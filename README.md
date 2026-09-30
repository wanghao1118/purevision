# PureVision：LIDC-IDRI 实例与跨数据集构造接口

本仓库提供 PureVision 在 LIDC-IDRI 肺结节 (pulmonary nodule) 数据集上的代码实例。内容覆盖 PureEyes 的局部病灶表型训练、整图表型训练、解剖训练，以及 PureNeurons 的共享语义对齐、病灶 patch 选择、软语义融合和冻结 MedGemma 1.5 解码。它是单一数据集、单一 backbone 的实例，不包含论文另外两个数据集和其他医学 VLM 的完整实验实现。

现已加入 CBIS-DDSM 乳腺病灶 (breast lesion) 与 3DReasonKnee 内侧半月板 (medial meniscus) 的数据构造配方和数据集自带类别规范。共享推理、报告解析及评测从构造后的 `dataset_contract.json` 读取解剖和表型维度，并核对对齐 checkpoint 的候选顺序；不会把三个数据集的类别写在共享推理代码里。原有完整训练仍是 LIDC 专用，不能把新的配方视为 CBIS/膝关节已完成三阶段训练。完整构造方法、服务器冻结清单哈希和论文差异见 [DATA_CONSTRUCTION_ZH.md](DATA_CONSTRUCTION_ZH.md)。

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

- 数据集 ID：`LIDC`；发布版：`lidc_purevision_r60_20260830_v1`；原始实验源根目录：`/datasets/LIDC`。
- 标签来源：LIDC-IDRI 放射科医师 XML 结节轮廓与表型评分、轮廓换算的物理直径，以及 TotalSegmentator 2.18.0 解剖伪标签。
- 患者级训练/验证/测试 (train/validation/test) 划分清单：`/datasets/LIDC/pathology_encoder_r28_full_fov/splits.json`；SHA-256：`3a22e64c084ba3327f3e73731a366385506a2e08e609d78c47f79182984de567`。
- 历史 backbone：`google/medgemma-1.5-4b-it`，snapshot `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`。本例的表型 R30、解剖 R39、共享对齐 R43 checkpoint 路径和 SHA-256 固定在 [`configs/07_medgemma15_reproduction_smoke.yaml`](configs/07_medgemma15_reproduction_smoke.yaml)。

数据集、完整权重和训练得到的 checkpoint 不在仓库内。运行时将本地数据与模型文件放到配置中的路径，或建立本地配置副本修改路径并更新对应哈希。`run_inference.py` 在加载前验证划分清单、backbone 索引与三个 checkpoint 的 SHA-256；不能只改路径而沿用不匹配的哈希。

### 类别中英对照

- 疾病：肺结节 (pulmonary nodule)。
- 解剖：体外 (outside body)、左肺 (left lung)、右肺 (right lung)、肺血管 (pulmonary vessel)、心脏 (heart)、骨 (bone)、外周软组织 (peripheral soft tissue)、左肺肺结节 (left-lung pulmonary nodule)、右肺肺结节 (right-lung pulmonary nodule)。
- 表型：密度 (density)、球形度 (sphericity)、边缘 (margin)、分叶 (lobulation)、毛刺 (spiculation)、钙化 (calcification)、大小 (size)。固定机器可读 ID 见 [`configs/05_alignment.yaml`](configs/05_alignment.yaml) 的 `text_targets`，显示名称与 ID 分开保存。

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

训练配置 `01` 至 `05` 描述可重新训练的阶段。真实病例测试的 `07` 配置绑定服务器上已存在的历史 R30/R39/R43 checkpoint；两套权重路径不可混用。更多目标函数和历史目录说明见 [`README_ZH.md`](README_ZH.md)。

## 真实病例测试

病例 `LIDC-IDRI-0079_s3_n0` 属于固定测试 (test) 划分。图像与病灶 mask 的 SHA-256、标签来源和参考类别见 [`case_zh.json`](examples/lidc_case_0079/case_zh.json)。mask 跨越 4×4 网格边界，参考单元按阳性像素最多的位置记录；它不满足论文正式 grounding 题的单格筛选条件。源数据的“左肺上部 (left upper lung)”是肺内高度标签，并非 4×4 网格标签。

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

该病例只是一例功能测试。此前运行中，PureVision 的解剖侧别 (anatomical side)、密度 (density)、毛刺 (spiculation) 和钙化 (calcification) 与冻结标签一致；球形度 (sphericity)、边缘 (margin)、分叶 (lobulation) 与大小 (size) 不一致。它不代表论文报告的 VQA 或 RRG 总体准确率。
当前脚本的新提问会明确要求 4×4 单元；仓库内 2026-09-30 的历史生成结果保持原样，使用的是旧提问文本，不能将两轮输出当作同一次无条件配对实验。

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

PowerShell 中设置密钥：`$env:OPENAI_API_KEY = 'your-api-key'`。模块记录模型、响应 ID、提示词与 schema 的哈希，以及原始解析输出，便于后续审计。论文未提供其 GPT6-Astra 固定解析提示词和运行输出，因此该模块提供可替换密钥的解析入口，不能视为论文解析结果的逐字复现。OpenAI 官方文档说明 GPT-6 Astra 支持 Responses API 与结构化输出：[模型文档](https://developers.openai.com/api/docs/models/gpt-6-astra)、[结构化输出文档](https://developers.openai.com/api/docs/guides/structured-outputs)。

## 复现范围

本仓库包含 CBIS-DDSM、3DReasonKnee 的构造代码、类别规范和可复算的评测构造/评分入口，但不包含其数据、训练 checkpoint、完整三阶段训练、多 backbone、论文原始 2,600 VQA/583 RRG 题目、其他方法对照或 GPT6-Astra 历史解析输出。提供的真实 LIDC 病例只能核验代码链路，不能替代论文表格数值复现。每份实验协议与结果文件都应固定数据集 ID、release、源根目录、标签来源以及 train/validation/test 清单和哈希。
