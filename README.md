# PureVision

本仓库提供以 LIDC-IDRI 肺结节 (pulmonary nodule) 为实例的 PureVision 代码，使用 MedGemma 1.5 作为 backbone。训练流程包括局部病灶表型、整图表型、解剖视觉编码器与共享语义对齐；推理流程包括病灶 patch 选择、软语义融合和冻结 decoder 解码。

CBIS-DDSM 乳腺病灶 (breast lesion) 与 3DReasonKnee 内侧半月板 (medial meniscus) 提供数据构造配方、数据集规范和评测入口。共享推理、报告解析及评测从构造后的 `dataset_contract.json` 读取解剖类别与表型维度。训练命令以下面的 LIDC-IDRI 实例为准。

仓库附带一例真实、去标识的 LIDC-IDRI 测试图像及结节 mask，可运行原生 MedGemma 1.5 与 PureVision 的配对测试。测试病例的图像、mask、冻结标签和既有生成记录位于 [`examples/lidc_case_0079/`](examples/lidc_case_0079/)。图像和 mask 供核验与离线评估；推理脚本不会把 mask 输入模型。

## 方法流程

1. 两个独立的 MedGemma 1.5 SigLIP 视觉塔分别学习表型 (phenotype) 与解剖 (anatomy) 表示。输入为 896×896，patch 为 14×14，输出 64×64 个 1,152 维 patch 表示。训练更新完整视觉塔，无 LoRA。
2. 表型塔先学习局部病灶几何，再学习包含上下文的完整图像几何。训练 mask 只用于选择受监督 patch。大小 (size) 使用训练集物理直径归一化后的连续目标；密度 (density)、球形度 (sphericity)、边缘 (margin)、分叶 (lobulation)、毛刺 (spiculation) 使用有序关系；钙化 (calcification) 使用分类关系。
3. 解剖塔学习正常组织、肺结节 (pulmonary nodule) 与父级肺区的几何结构，并对肺血管 (pulmonary vessel) 这一易混淆结构使用 teacher-relative 约束。
4. 冻结双视觉塔和语言模型，仅训练共享 RMSNorm 与无偏置 1,152→2,560 投影。文本目标由冻结语言模型编码。
5. 推理时对每个解剖 patch 计算“最佳病灶 cosine 减最佳其他解剖 cosine”；在最符合病灶的 5×5 邻域内选 Top-8 patch，并对病灶分数 softmax 得空间权重。
6. 解剖 (anatomy) 与七个表型组分别聚合候选 cosine，组内标准化后以温度 0.125 取得软类别概率。候选文本 token 序列通过重复末 token 补齐，再按概率融合。
7. 保留原生视觉 token，在其后、任务指令前插入估计位置和融合 token，由冻结 MedGemma 1.5 decoder 生成文本。推理不使用 mask，也不执行训练期 mask-conditioned pooling、监督质心评估或 hard-centroid 分类。

本仓库不生成 t-SNE 图；若另行绘制，只能将其视为二维可视化。表征的定量距离须在原始归一化特征空间计算，不能用 t-SNE 图上的簇间空隙代替。

## 数据与权重

冻结数据中文索引版本：`2026-09-30-v1`。新构造应生成独立 release 与清单哈希。

| 数据集 ID / release | 源根目录 | 标签来源 | 患者级 train/val/test 清单及 SHA-256 |
|---|---|---|---|
| `LIDC` / `lidc_purevision_r60_20260830_v1` | `/datasets/LIDC` | 医师 XML 结节轮廓、表型评分和物理直径；TotalSegmentator 2.18.0 解剖伪标签 | `/datasets/LIDC/pathology_encoder_r28_full_fov/splits.json`；`3a22e64c084ba3327f3e73731a366385506a2e08e609d78c47f79182984de567` |
| `CBIS-DDSM` / `cbis_native_four_part_v2_20260918` | `/mnt/sda/hao/wh/datasets/CBIS-DDSM` | 原始 DICOM 病灶 ROI、GrabCut 乳腺组织 (breast tissue)、Attention U-Net 胸肌 (pectoral muscle) 伪标签及源 CSV 表型 | `/datasets/cbis_ddsm/image_lesion_anatomy_text_native_v2_20260918/manifest.jsonl`；`81991e04e7a5cfc50f54614c66e2e30417c3ab2985e2813b527d42c61b5f9b62` |
| `3DReasonKnee` / `medial_meniscus_strict2d_crop128_v4` | `/datasets/3dreasonknee` | OAI 膝关节 MRI、检查级区域 MOAKS 内侧半月板外突等级 (medial meniscus medial extrusion grade)、人工优先的解剖掩码及模型补全通道；整块内侧半月板代理 ROI 用于定位参考 | `/datasets/3dreasonknee/medial_meniscus_strict2d_crop128_v4/manifest/all.jsonl`；`6fe7bc1aad69e79fa3103ba0ffaa5d832bfac2309b629fe701d4bc73343b8d51` |

历史 backbone 为 `google/medgemma-1.5-4b-it`，snapshot `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`。LIDC 实例的表型 R30、解剖 R39、共享对齐 R43 checkpoint 路径和 SHA-256 位于 [`configs/07_medgemma15_reproduction_smoke.yaml`](configs/07_medgemma15_reproduction_smoke.yaml)。

数据集、完整权重和训练得到的 checkpoint 不在仓库内。运行时将本地数据与模型文件放到配置中的路径，或建立本地配置副本修改路径并更新对应哈希。`run_inference.py` 在加载前验证划分清单、backbone 索引与三个 checkpoint 的 SHA-256；不能只改路径而沿用不匹配的哈希。

### 类别中英对照

- 疾病：肺结节 (pulmonary nodule)。
- 解剖：体外 (outside body)、左肺 (left lung)、右肺 (right lung)、肺血管 (pulmonary vessel)、心脏 (heart)、骨 (bone)、外周软组织 (peripheral soft tissue)、左肺肺结节 (left-lung pulmonary nodule)、右肺肺结节 (right-lung pulmonary nodule)。
- 表型：密度 (density)、球形度 (sphericity)、边缘 (margin)、分叶 (lobulation)、毛刺 (spiculation)、钙化 (calcification)、大小 (size)。固定机器可读 ID 见 [`configs/05_alignment.yaml`](configs/05_alignment.yaml) 的 `text_targets`，显示名称与 ID 分开保存。

## 数据构造与评测

以下命令在具有相应原始数据和依赖的服务器执行；构造产物与评测实例应写到仓库外。每次新构造都应保存 `dataset_contract.json` 以及实际划分清单与 SHA-256。

```bash
PYTHONPATH=src python -m purevision.preprocess --config configs/01_phenotype_local.yaml --output-dir /datasets/LIDC/new_lung_crop_intermediate
PYTHONPATH=src python -m purevision.splits --config configs/01_phenotype_local.yaml --processed-dir /datasets/LIDC/new_lung_crop_intermediate
PYTHONPATH=src python scripts/build_lidc_full_fov.py --source /datasets/LIDC/new_lung_crop_intermediate --output /datasets/LIDC/new_full_fov_release
PYTHONPATH=src python scripts/build_dataset_contract.py --recipe dataset_recipes/lidc.yaml --dataset-root /datasets/LIDC/new_full_fov_release
```

```bash
python dataset_construction/build_cbis_native_four_part_dataset.py --source-root /mnt/sda/hao/wh/datasets/CBIS-DDSM --v1-root /datasets/cbis_ddsm/image_lesion_anatomy_text_v1 --revision-root /datasets/cbis_ddsm/native_revision_v1 --output /datasets/cbis_ddsm/new_native_release
PYTHONPATH=src python scripts/build_dataset_contract.py --recipe dataset_recipes/cbis.yaml --dataset-root /datasets/cbis_ddsm/new_native_release
```

```bash
python dataset_construction/build_knee_strict2d_dataset.py prepare --input /datasets/3dreasonknee/strict2d_source --output /datasets/3dreasonknee/new_v4 --model /models/coronal_best_model.h5 --pilot /audits/knee_45_case_pilot
python dataset_construction/build_knee_strict2d_dataset.py build --output /datasets/3dreasonknee/new_v4
python dataset_construction/build_knee_strict2d_dataset.py verify --output /datasets/3dreasonknee/new_v4
PYTHONPATH=src python scripts/build_dataset_contract.py --recipe dataset_recipes/knee.yaml --dataset-root /datasets/3dreasonknee/new_v4
```

CBIS 构造需要历史 v1 样本与胸肌修订产物；膝关节构造需要源 MOAKS 清单、pilot 和冻结模型。评测读取数据集自带类别，膝关节以代理 ROI 的阳性像素最多网格为定位答案；二类或三类表型使用实际候选数。对构造出的规范运行：

```bash
PYTHONPATH=src python scripts/validate_dataset_contract.py --dataset-contract /datasets/<release>/dataset_contract.json
PYTHONPATH=src python scripts/build_benchmark.py --dataset-contract /datasets/<release>/dataset_contract.json --output-dir /benchmarks/<release> --phenotype-questions 200 --grounding-questions 200 --rrg-cases 200
```

3DReasonKnee 的评测题数使用 `--phenotype-questions 100 --grounding-questions 100 --rrg-cases 183`。参考清单、题目和模型预测由 [`scripts/score_benchmark.py`](scripts/score_benchmark.py) 评分。

## 环境与训练

在装有 PyTorch/CUDA 的 Python 3.10+ 环境中运行；真实病例测试所用环境为 Python 3.12、PyTorch 2.11.0+cu130、Transformers 5.5.0。安装项目依赖：

```bash
python -m pip install -e '.[preprocess,test]'
python -m pytest -q
```

按以下顺序执行 LIDC-IDRI 实例的训练与推理。训练需要配置中指定的数据清单和 MedGemma 1.5 权重：

```bash
torchrun --nproc_per_node=2 -m purevision.train --config configs/01_phenotype_local.yaml
torchrun --nproc_per_node=2 -m purevision.train --config configs/02_phenotype_global.yaml --initialize-from runs/phenotype_local/best.pt
python scripts/train_anatomy_base.py --config configs/03_anatomy_base.yaml
python scripts/train_anatomy.py --config configs/04_anatomy_final.yaml --initialize-from runs/anatomy_base/best.pt
python scripts/extract_alignment_features.py --config configs/05_alignment.yaml --kind all
python scripts/train_alignment.py --config configs/05_alignment.yaml
```

训练配置 `01` 至 `05` 用于训练；`07` 配置绑定历史 R30/R39/R43 checkpoint，用于真实病例测试。

## 真实病例测试

病例 `LIDC-IDRI-0079_s3_n0` 属于固定测试 (test) 划分。图像与病灶 mask 的 SHA-256、标签来源和参考类别见 [`case_zh.json`](examples/lidc_case_0079/case_zh.json)。参考 4×4 网格单元按 mask 阳性像素最多的位置记录。

无需模型权重即可先核验真实样本与 mask：

```bash
PYTHONPATH=src python scripts/run_real_case.py --check-only
```

备齐 `07` 配置绑定的数据划分与权重后，执行配对生成。下例使用物理 GPU1，进程内呈现为 `cuda:0`：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python scripts/run_real_case.py \
  --config configs/07_medgemma15_reproduction_smoke.yaml \
  --output-dir runs/lidc_case_0079
```

输出包括 `native_result_zh.json`、`purevision_result_zh.json`、`raw_patch_embeddings.pt` 和 `SUMMARY_ZH.json`。原生分支使用 MedGemma 1.5 原始预训练权重的标准未修改推理；PureVision 分支加载冻结的双塔和共享对齐器。原始双塔特征在共享对齐器之前保存，供复核。仓库附带的 [`SUMMARY_ZH.json`](examples/lidc_case_0079/SUMMARY_ZH.json) 和两份 `*_result_zh.json` 是 2026-09-30 使用仓库内图像完成的真实运行记录。mask 阳性像素分别落在 `r3c3` 66 个、`r4c3` 145 个，因此单格筛选为 `false`。原始特征二进制未纳入代码仓库，其 SHA-256 已写在结果记录中。

该病例用于单例功能测试。既有生成记录保持原样，供检查原生 MedGemma 1.5 与 PureVision 两条推理链路。

## GPT6-Astra 报告解析

[`src/purevision/rrg_parser.py`](src/purevision/rrg_parser.py) 使用 GPT6-Astra 的 Responses API 结构化输出，将自由文本报告映射到固定的 4×4 网格和数据集自带的表型机器 ID。历史 LIDC 实例仍可从 `configs/05_alignment.yaml` 读取词表；三数据集统一入口改用 `--dataset-contract`。缺失、冲突或无效字段返回 `null`。解析器仅抽取标签，评分由冻结参考和确定性指标另行计算。

只验证仓库附带的既有生成报告时，无需下载 MedGemma 权重。安装 API 客户端后，在自己的环境中设置密钥，仓库中不填写密钥：

```bash
python -m pip install 'openai>=2.0' 'PyYAML>=6.0'
export OPENAI_API_KEY='your-api-key'
PYTHONPATH=src python scripts/parse_rrg_report.py \
  --config configs/07_medgemma15_reproduction_smoke.yaml \
  --label-config configs/05_alignment.yaml \
  --result-json examples/lidc_case_0079/purevision_result_zh.json \
  --output runs/lidc_case_0079/parsed_rrg_zh.json
```

PowerShell 中设置密钥：`$env:OPENAI_API_KEY = 'your-api-key'`。模块记录模型、响应 ID、提示词与 schema 的哈希，以及解析输出。数据和 checkpoint 不随代码发布。
