# PureVision

PureVision is a MedGemma 1.5-based medical vision-language baseline. This repository provides the LIDC-IDRI pulmonary nodule training example, from local and full-image phenotype learning through anatomy learning, shared semantic alignment, patch selection, soft token fusion, and frozen-decoder generation.

Dataset construction recipes and evaluation tools are also provided for CBIS-DDSM breast lesions and 3DReasonKnee medial meniscus injury. Shared inference, report parsing, and evaluation read anatomy and phenotype definitions from each constructed dataset's `dataset_contract.json`. The training commands below describe the LIDC-IDRI example.

One de-identified LIDC-IDRI test case, its nodule mask, labels, and archived generation records are in [`examples/lidc_case_0079/`](examples/lidc_case_0079/). The mask is used for verification and evaluation, never as a model input at inference time.

## Method

1. Two independent MedGemma 1.5 SigLIP vision towers learn phenotype and anatomy representations. A 896 x 896 image yields a 64 x 64 grid of 1,152-dimensional patch features with 14 x 14 patches. Both towers are fully fine-tuned without LoRA.
2. The phenotype tower learns local lesion geometry, then full-image features with surrounding context. During training, the lesion mask selects supervised patches. Size uses a continuous target derived from physical diameter; density, sphericity, margin, lobulation, and spiculation use ordinal relations; calcification uses categorical relations.
3. The anatomy tower learns tissue, nodule, and parent-lung structure. A teacher-relative constraint handles pulmonary vessels as a potential confounder.
4. Both vision towers and the language model are frozen for alignment. Only a shared RMSNorm and bias-free 1,152-to-2,560 projection are trained against text targets encoded by the frozen language model.
5. Inference scores each patch by its best lesion-anatomy cosine similarity minus its best non-lesion-anatomy similarity. It selects the top eight patches within the highest-scoring 5 x 5 neighborhood and applies a softmax over lesion scores.
6. Anatomy and phenotype candidate scores are normalized within each group, converted to soft probabilities at temperature 0.125, and used to fuse candidate text tokens. Shorter token sequences are padded by repeating their final token.
7. The fused tokens and estimated location are inserted after the native visual tokens and before the task instruction. The MedGemma 1.5 decoder remains frozen. Inference does not use masks or training-time mask-conditioned pooling.

The repository does not generate t-SNE plots. Quantitative feature distances should be computed in the original normalized embedding space, not from gaps in a two-dimensional visualization.

## Data and Weights

The following is the versioned index for the frozen dataset examples (`2026-09-30-v1`). New constructions need their own release identifier and split-manifest hash. Paths document source provenance; they do not define dataset identity by themselves.

| Dataset ID / release | Source root | Label provenance | Patient-level train/val/test manifest and SHA-256 |
|---|---|---|---|
| `LIDC` / `lidc_purevision_r60_20260830_v1` | `/datasets/LIDC` | Radiologist XML nodule contours and phenotype ratings, physical diameter, and TotalSegmentator 2.18.0 anatomy pseudo-labels | `/datasets/LIDC/pathology_encoder_r28_full_fov/splits.json`; `3a22e64c084ba3327f3e73731a366385506a2e08e609d78c47f79182984de567` |
| `CBIS-DDSM` / `cbis_native_four_part_v2_20260918` | `/mnt/sda/hao/wh/datasets/CBIS-DDSM` | Source DICOM lesion ROIs, GrabCut breast-tissue masks, Attention U-Net pectoral-muscle pseudo-labels, and source CSV phenotypes | `/datasets/cbis_ddsm/image_lesion_anatomy_text_native_v2_20260918/manifest.jsonl`; `81991e04e7a5cfc50f54614c66e2e30417c3ab2985e2813b527d42c61b5f9b62` |
| `3DReasonKnee` / `medial_meniscus_strict2d_crop128_v4` | `/datasets/3dreasonknee` | OAI knee MRI, exam-level regional MOAKS medial-meniscus-extrusion grades, anatomy masks with model-filled channels, and a whole-medial-meniscus ROI used as the grounding reference | `/datasets/3dreasonknee/medial_meniscus_strict2d_crop128_v4/manifest/all.jsonl`; `6fe7bc1aad69e79fa3103ba0ffaa5d832bfac2309b629fe701d4bc73343b8d51` |

The historical backbone is `google/medgemma-1.5-4b-it`, snapshot `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`. Paths and SHA-256 hashes for the LIDC phenotype R30, anatomy R39, and shared-alignment R43 checkpoints are recorded in [`configs/07_medgemma15_reproduction_smoke.yaml`](configs/07_medgemma15_reproduction_smoke.yaml). Full datasets and checkpoints are not included. The inference entry point verifies the split manifest, backbone index, and checkpoint hashes before loading.

The LIDC disease category is pulmonary nodule. Its anatomy categories are outside body, left lung, right lung, pulmonary vessel, heart, bone, peripheral soft tissue, left-lung pulmonary nodule, and right-lung pulmonary nodule. Its phenotype dimensions are density, sphericity, margin, lobulation, spiculation, calcification, and size. Stable machine-readable IDs and text targets are in [`configs/05_alignment.yaml`](configs/05_alignment.yaml); other datasets supply their own categories through their contracts.

## Dataset Construction and Evaluation

Run construction on a machine with the corresponding source data and dependencies. Keep constructed datasets and benchmark instances outside this repository. Each construction should retain its `dataset_contract.json`, split manifests, and hashes.

LIDC-IDRI:

```bash
PYTHONPATH=src python -m purevision.preprocess --config configs/01_phenotype_local.yaml --output-dir /datasets/LIDC/new_lung_crop_intermediate
PYTHONPATH=src python -m purevision.splits --config configs/01_phenotype_local.yaml --processed-dir /datasets/LIDC/new_lung_crop_intermediate
PYTHONPATH=src python scripts/build_lidc_full_fov.py --source /datasets/LIDC/new_lung_crop_intermediate --output /datasets/LIDC/new_full_fov_release
PYTHONPATH=src python scripts/build_dataset_contract.py --recipe dataset_recipes/lidc.yaml --dataset-root /datasets/LIDC/new_full_fov_release
```

CBIS-DDSM requires the prior v1 sample metadata and pectoral-muscle revision assets:

```bash
python dataset_construction/build_cbis_native_four_part_dataset.py --source-root /mnt/sda/hao/wh/datasets/CBIS-DDSM --v1-root /datasets/cbis_ddsm/image_lesion_anatomy_text_v1 --revision-root /datasets/cbis_ddsm/native_revision_v1 --output /datasets/cbis_ddsm/new_native_release
PYTHONPATH=src python scripts/build_dataset_contract.py --recipe dataset_recipes/cbis.yaml --dataset-root /datasets/cbis_ddsm/new_native_release
```

3DReasonKnee requires the prepared source subset, MOAKS labels, pilot records, and frozen segmentation model:

```bash
python dataset_construction/build_knee_strict2d_dataset.py prepare --input /datasets/3dreasonknee/strict2d_source --output /datasets/3dreasonknee/new_v4 --model /models/coronal_best_model.h5 --pilot /audits/knee_45_case_pilot
python dataset_construction/build_knee_strict2d_dataset.py build --output /datasets/3dreasonknee/new_v4
python dataset_construction/build_knee_strict2d_dataset.py verify --output /datasets/3dreasonknee/new_v4
PYTHONPATH=src python scripts/build_dataset_contract.py --recipe dataset_recipes/knee.yaml --dataset-root /datasets/3dreasonknee/new_v4
```

Validate a constructed contract, then create the test questions and RRG reference set:

```bash
PYTHONPATH=src python scripts/validate_dataset_contract.py --dataset-contract /datasets/<release>/dataset_contract.json
PYTHONPATH=src python scripts/build_benchmark.py --dataset-contract /datasets/<release>/dataset_contract.json --output-dir /benchmarks/<release> --phenotype-questions 200 --grounding-questions 200 --rrg-cases 200
```

For 3DReasonKnee, use `--phenotype-questions 100 --grounding-questions 100 --rrg-cases 183`. Its grounding answer is the grid cell containing the most positive pixels in the reference ROI. Phenotype questions use the candidate count provided by the dataset contract, up to four options. Use [`scripts/score_benchmark.py`](scripts/score_benchmark.py) to score predictions against the generated references.

## Installation and LIDC Training

Use Python 3.10+ with PyTorch/CUDA and a local copy of the MedGemma 1.5 weights. The included real-case run used Python 3.12, PyTorch 2.11.0+cu130, and Transformers 5.5.0.

```bash
python -m pip install -e '.[preprocess,construction,parser,test]'
python -m pytest -q
```

Run the LIDC training stages in order:

```bash
torchrun --nproc_per_node=2 -m purevision.train --config configs/01_phenotype_local.yaml
torchrun --nproc_per_node=2 -m purevision.train --config configs/02_phenotype_global.yaml --initialize-from runs/phenotype_local/best.pt
python scripts/train_anatomy_base.py --config configs/03_anatomy_base.yaml
python scripts/train_anatomy.py --config configs/04_anatomy_final.yaml --initialize-from runs/anatomy_base/best.pt
python scripts/extract_alignment_features.py --config configs/05_alignment.yaml --kind all
python scripts/train_alignment.py --config configs/05_alignment.yaml
```

Configs `01` through `05` describe training stages. Config `07` binds the historical R30/R39/R43 checkpoints for the real-case run.

## Real LIDC-IDRI Test Case

`LIDC-IDRI-0079_s3_n0` is in the frozen test split. Its image and mask hashes, label provenance, and reference categories are in [`case_zh.json`](examples/lidc_case_0079/case_zh.json). The reference 4 x 4 grid cell is determined by the largest count of positive mask pixels.

Verify the case without model weights:

```bash
PYTHONPATH=src python scripts/run_real_case.py --check-only
```

With the data and weights specified by config `07`, run native MedGemma 1.5 inference and PureVision on the same case. The example selects physical GPU 1, exposed inside the process as `cuda:0`:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python scripts/run_real_case.py \
  --config configs/07_medgemma15_reproduction_smoke.yaml \
  --output-dir runs/lidc_case_0079
```

The run writes `native_result_zh.json`, `purevision_result_zh.json`, `raw_patch_embeddings.pt`, and `SUMMARY_ZH.json`. The native branch uses standard, unmodified inference with the original pretrained weights. The PureVision branch loads the frozen, trained vision towers and shared aligner. Raw tower embeddings are saved before alignment. Archived case outputs are included in [`examples/lidc_case_0079/`](examples/lidc_case_0079/); the binary embedding file is not committed.

## GPT6-Astra Report Parser

[`src/purevision/rrg_parser.py`](src/purevision/rrg_parser.py) maps free-text reports to a 4 x 4 location and the phenotype IDs supplied by the dataset contract. It extracts labels only; benchmark scoring is deterministic and separate. Set your own `OPENAI_API_KEY` locally:

```bash
export OPENAI_API_KEY='your-api-key'
PYTHONPATH=src python scripts/parse_rrg_report.py \
  --config configs/07_medgemma15_reproduction_smoke.yaml \
  --label-config configs/05_alignment.yaml \
  --result-json examples/lidc_case_0079/purevision_result_zh.json \
  --output runs/lidc_case_0079/parsed_rrg_zh.json
```

In PowerShell: `$env:OPENAI_API_KEY = 'your-api-key'`. For another dataset, use `--dataset-contract` to read its phenotype vocabulary. No API key is stored in this repository.
