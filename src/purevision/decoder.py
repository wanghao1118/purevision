from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


@dataclass(frozen=True)
class DecoderPrompt:
    input_ids: torch.Tensor
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor
    native_positions: torch.Tensor
    semantic_positions: torch.Tensor


def evidence_prefix(
    *,
    center_row: float,
    center_column: float,
    grid_size: int,
    group_token_counts: Mapping[str, int],
    image_token: str,
) -> str:
    if grid_size <= 0 or not image_token:
        raise ValueError("grid_size and image_token must be valid")
    parts = [
        (
            "Estimated lesion location on the encoder grid: "
            f"row {center_row:.3f} of {grid_size}, "
            f"column {center_column:.3f} of {grid_size}.\n"
        )
    ]
    for group, count in group_token_counts.items():
        if int(count) <= 0:
            raise ValueError(f"semantic group {group} has no soft tokens")
        parts.append(f"{group}: {image_token * int(count)}\n")
    return "".join(parts)


def replace_visual_embeddings(
    inputs_embeds: torch.Tensor,
    *,
    native_positions: torch.Tensor,
    native_tokens: torch.Tensor,
    semantic_positions: torch.Tensor,
    semantic_tokens: torch.Tensor,
) -> torch.Tensor:
    if inputs_embeds.ndim != 3 or inputs_embeds.shape[0] != 1:
        raise ValueError("inputs_embeds must have shape [1, sequence, dimension]")
    dimension = inputs_embeds.shape[-1]
    native_positions = native_positions.long().reshape(-1)
    semantic_positions = semantic_positions.long().reshape(-1)
    if native_tokens.shape != (len(native_positions), dimension):
        raise ValueError("native visual tokens do not match their prompt positions")
    if semantic_tokens.shape != (len(semantic_positions), dimension):
        raise ValueError("semantic soft tokens do not match their prompt positions")
    if set(native_positions.tolist()) & set(semantic_positions.tolist()):
        raise ValueError("native and semantic prompt positions overlap")
    result = inputs_embeds.clone()
    result[0, native_positions.to(result.device)] = native_tokens.to(
        device=result.device, dtype=result.dtype
    )
    result[0, semantic_positions.to(result.device)] = semantic_tokens.to(
        device=result.device, dtype=result.dtype
    )
    return result


class FrozenMedGemmaDecoder:


    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str | torch.device = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        attention_implementation: str = "sdpa",
    ) -> None:
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, local_files_only=True
        )
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_path,
            local_files_only=True,
            dtype=dtype,
            attn_implementation=attention_implementation,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.requires_grad_(False).eval()
        self.native_token_count = int(
            getattr(self.processor, "image_seq_length", 256)
        )
        self.image_token_id = int(self.tokenizer.image_token_id)
        image_token = getattr(self.tokenizer, "image_token", None)
        self.image_token = (
            str(image_token)
            if image_token
            else str(self.tokenizer.convert_ids_to_tokens(self.image_token_id))
        )

    def _autocast(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=self.dtype)
        return nullcontext()

    @torch.inference_mode()
    def preprocess_image(self, image: Any) -> torch.Tensor:
        encoded = self.processor.image_processor(images=[image], return_tensors="pt")
        return encoded["pixel_values"].to(self.device)

    @torch.inference_mode()
    def native_visual_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with self._autocast():
            tokens = self.model.model.get_image_features(
                pixel_values=pixel_values
            ).pooler_output
        expected = (
            pixel_values.shape[0],
            self.native_token_count,
            self.model.get_input_embeddings().weight.shape[1],
        )
        if tuple(tokens.shape) != expected:
            raise RuntimeError(
                f"unexpected native visual token shape {tuple(tokens.shape)}; "
                f"expected {expected}"
            )
        return tokens

    @torch.inference_mode()
    def candidate_token_sequences(
        self, texts: Sequence[str]
    ) -> list[torch.Tensor]:
        embedding = self.model.get_input_embeddings()
        output = []
        for text in texts:
            ids = self.tokenizer(
                str(text), add_special_tokens=False, return_tensors="pt"
            )["input_ids"][0]
            if not len(ids):
                raise ValueError(f"candidate text produced no tokens: {text!r}")
            output.append(embedding(ids.to(self.device)).detach().float())
        return output

    def _formatted_prompt(self, prefix: str, instruction: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{prefix}\n{instruction}"},
                ],
            }
        ]
        formatted = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if formatted.count(self.processor.boi_token) != 1:
            raise ValueError("expected exactly one native image marker")
        return formatted.replace(
            self.processor.boi_token, self.processor.full_image_sequence
        )

    @torch.inference_mode()
    def prepare_prompt(
        self,
        pixel_values: torch.Tensor,
        semantic_groups: Mapping[str, torch.Tensor],
        *,
        center_row: float,
        center_column: float,
        grid_size: int,
        instruction: str,
    ) -> DecoderPrompt:
        if not instruction.strip() or not semantic_groups:
            raise ValueError("instruction and semantic groups must be non-empty")
        counts = {name: len(tokens) for name, tokens in semantic_groups.items()}
        prefix = evidence_prefix(
            center_row=center_row,
            center_column=center_column,
            grid_size=grid_size,
            group_token_counts=counts,
            image_token=self.image_token,
        )
        formatted = self._formatted_prompt(prefix, instruction)
        input_ids = self.tokenizer(formatted, return_tensors="pt")["input_ids"]
        positions = torch.where(input_ids[0] == self.image_token_id)[0]
        semantic_tokens = torch.cat(tuple(semantic_groups.values()), dim=0)
        expected = self.native_token_count + len(semantic_tokens)
        if len(positions) != expected:
            raise ValueError(
                f"expected {expected} image placeholders, found {len(positions)}"
            )
        native_positions = positions[: self.native_token_count]
        semantic_positions = positions[self.native_token_count :]
        token_type_ids = torch.tensor(
            self.processor.create_mm_token_type_ids([input_ids[0].tolist()]),
            dtype=torch.long,
        )
        token_type_ids[0, semantic_positions] = 0
        input_ids = input_ids.to(self.device)
        attention_mask = torch.ones_like(input_ids)
        token_type_ids = token_type_ids.to(self.device)
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        native_tokens = self.native_visual_tokens(pixel_values)[0]
        inputs_embeds = replace_visual_embeddings(
            inputs_embeds,
            native_positions=native_positions,
            native_tokens=native_tokens,
            semantic_positions=semantic_positions,
            semantic_tokens=semantic_tokens,
        )
        return DecoderPrompt(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            native_positions=native_positions,
            semantic_positions=semantic_positions,
        )

    @torch.inference_mode()
    def generate(
        self,
        pixel_values: torch.Tensor,
        semantic_groups: Mapping[str, torch.Tensor],
        *,
        center_row: float,
        center_column: float,
        grid_size: int,
        instruction: str,
        max_new_tokens: int = 384,
    ) -> str:
        prompt = self.prepare_prompt(
            pixel_values,
            semantic_groups,
            center_row=center_row,
            center_column=center_column,
            grid_size=grid_size,
            instruction=instruction,
        )
        generated = self.model.generate(
            input_ids=None,
            inputs_embeds=prompt.inputs_embeds,
            attention_mask=prompt.attention_mask,
            token_type_ids=prompt.token_type_ids,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.model.generation_config.eos_token_id,
        )
        return self.tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
