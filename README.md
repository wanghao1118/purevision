# PureVision: Geometry-Supervised Visual Representation Learning for Multi-Phenotype Lesion Interpretation in Medical VLMs

PureVision learns anatomy- and phenotype-aware visual representations for lesion interpretation in medical vision-language models. **PureEyes** trains the visual encoders, while **PureNeurons** selects lesion evidence and delivers it to a frozen decoder.

## Overview

This repository provides a MedGemma 1.5 implementation with a LIDC-IDRI pulmonary nodule training example. It also includes data construction recipes and dataset-driven evaluation tools for CBIS-DDSM and 3DReasonKnee. Anatomy and phenotype categories are read from each dataset's `dataset_contract.json`.

![PureVision method overview](assets/figures/method.png)

*PureEyes structures anatomical and phenotypic representations; PureNeurons selects and fuses lesion-relevant evidence.*

## Results

The following figures are from the accompanying manuscript. The anatomy figure includes anatomy-text cosine similarities; the t-SNE panels are qualitative visualizations, not quantitative embedding-distance measurements.

![Anatomy representation and anatomy-text alignment analysis](assets/figures/anatomy_analysis.png)

*Anatomical representations and anatomy-text alignment on LIDC-IDRI.*

![Phenotype representation analysis](assets/figures/phenotype_analysis.png)

*Visualizations of nodule size, density, calcification, and spiculation representations on LIDC-IDRI.*

## Qualitative Analysis

![Lesion patch selection examples](assets/figures/patch_selection.png)

*Examples of lesion-focused patch selection by PureNeurons.*

![PureVision manuscript case study](assets/figures/case_study.png)

*Manuscript case study showing the lesion mask, representation views, selected patches, and generated report. The runnable test case below is a separate example.*

## Quick Start

```bash
git clone https://github.com/wanghao1118/purevision.git
cd purevision
python -m pip install -e '.[preprocess,construction,parser,test]'
python -m pytest -q
```

Place your datasets and MedGemma 1.5 weights outside the repository, then update the local configuration paths. Dataset builders are in [`dataset_construction/`](dataset_construction/), with recipes in [`dataset_recipes/`](dataset_recipes/). Use [`scripts/build_dataset_contract.py`](scripts/build_dataset_contract.py) to create a contract from a constructed dataset.

## Training

The LIDC-IDRI pipeline uses configs `01` through `05` for local phenotype training, full-image phenotype training, anatomy training, and shared alignment. Run the stages in order:

```bash
torchrun --nproc_per_node=2 -m purevision.train --config configs/01_phenotype_local.yaml
torchrun --nproc_per_node=2 -m purevision.train --config configs/02_phenotype_global.yaml --initialize-from runs/phenotype_local/best.pt
python scripts/train_anatomy_base.py --config configs/03_anatomy_base.yaml
python scripts/train_anatomy.py --config configs/04_anatomy_final.yaml --initialize-from runs/anatomy_base/best.pt
python scripts/extract_alignment_features.py --config configs/05_alignment.yaml --kind all
python scripts/train_alignment.py --config configs/05_alignment.yaml
```

## Test

The repository includes a separate de-identified LIDC-IDRI [test case](examples/lidc_case_0079/) with an image and lesion mask. Check its files without model weights:

```bash
PYTHONPATH=src python scripts/run_real_case.py --check-only
```

With the weights and data paths specified in [`configs/07_medgemma15_reproduction_smoke.yaml`](configs/07_medgemma15_reproduction_smoke.yaml), run native MedGemma 1.5 and PureVision on the same case:

```bash
PYTHONPATH=src python scripts/run_real_case.py --config configs/07_medgemma15_reproduction_smoke.yaml
```

For dataset-level evaluation, use [`scripts/build_benchmark.py`](scripts/build_benchmark.py) and [`scripts/score_benchmark.py`](scripts/score_benchmark.py) with a dataset contract.

## Report Parsing

[`src/purevision/rrg_parser.py`](src/purevision/rrg_parser.py) provides a GPT6-Astra report parser. Set your own `OPENAI_API_KEY` and run [`scripts/parse_rrg_report.py`](scripts/parse_rrg_report.py) with a dataset contract and generated report. No API key is stored in this repository.
