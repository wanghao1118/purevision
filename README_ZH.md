# PureVision 论文主流程代码

本目录是从 MEBE 工程中整理出的 LIDC-IDRI 实例代码副本，主链截止到 R60：局部病灶表型视觉编码器训练、完整图像表型视觉编码器训练、完整图像解剖视觉编码器训练、共享语义对齐、局部病灶选择、soft token fusion 和冻结 decoder 解码。仓库附带一例去标识的真实测试图像、病灶 mask 和生成记录；完整数据、权重、embedding 与缓存不在仓库内。新读者从 [README.md](README.md) 开始。

## 范围

1. `01_phenotype_local.yaml`：局部病灶表型视觉编码器（local lesion phenotype vision encoder）。
2. `02_phenotype_global.yaml`：从局部阶段初始化，在完整图像上训练全局表型视觉编码器（global phenotype vision encoder）。mask 只选择训练期监督 patch。
3. `03_anatomy_base.yaml` 与 `04_anatomy_final.yaml`：独立初始化并训练完整图像解剖视觉编码器（global anatomy vision encoder）。
4. `05_alignment.yaml`：冻结两套视觉编码器，只训练共享 `RMSNorm + bias-free 1152 -> 2560` 对齐器。
5. `06_inference.yaml`：按 R60 主方法完成 5x5 局部选择、top-8 patch、概率加权语义融合和冻结 MedGemma decoder 解码。

R61-R73 的推理增强、跨 backbone、peer baseline、消融、hard centroid 和 t-SNE 交付代码均不属于本副本。`hard_centroid` 不是主方法；t-SNE 也不参与模型推理或定量距离计算。

## 类别中英对照

疾病：肺结节 / pulmonary nodule。

解剖类别：体外 / outside body、左肺 / left lung、右肺 / right lung、肺血管 / pulmonary vessel、心脏 / heart、骨 / bone、外周软组织 / peripheral soft tissue、左肺肺结节 / left-lung pulmonary nodule、右肺肺结节 / right-lung pulmonary nodule。

表型维度：密度 / density、球形度 / sphericity、边缘 / margin、分叶 / lobulation、毛刺 / spiculation、钙化 / calcification、大小 / size。

表型候选：非实性或主要非实性 / non-solid or mostly non-solid、混合或主要实性 / mixed or mostly solid、实性 / solid；线性或主要线性 / linear or mostly linear、卵圆 / ovoid、主要圆形 / mostly round、圆形 / round；边界不清 / poorly defined、接近边界不清 / nearly poorly defined、中等边缘 / intermediate margin、接近锐利 / nearly sharp、锐利 / sharp；无分叶 / no lobulation、分叶 / lobulation；无毛刺 / no spiculation、毛刺 / spiculation；无钙化 / no calcification、钙化 / calcification；3-6 毫米 / 3 to 6 mm、6-10 毫米 / 6 to 10 mm、大于 10 毫米 / greater than 10 mm。

机器可读 ID（如 `left_lung`、`spiculation`）保持不变。

## 与论文一致的修正

- 两个视觉编码器是互相独立的完整 SigLIP tower，输入 896x896、patch 14、网格 64x64、视觉维度 1152；不是 LoRA。
- 表型训练使用 AdamW、weight decay 0.01、5% warm-up 后 cosine、gradient clip 1、BF16、学习率 `5e-7`、30 epochs、global batch 16。
- 所有表型距离均基于 L2 归一化向量的欧氏距离，Smooth-L1 的 `delta=1`。大小 / size 使用训练集物理直径 min-max 连续值和 `1.5 * |delta size|`，不再按三档序数标签训练几何。
- 表型 memory bank 仅保存最近 1024 个 detached 病灶表示。七维权重依次为 0.125、0.125、0.125、0.25、0.1875、0.09375、0.09375。
- 解剖总目标固定为 `4*L_anatomy + 0.55*L_parent + 4*L_hierarchy + 10*L_distill + L_conf`；非病灶蒸馏 teacher 是冻结的原始 MedGemma 视觉 tower；肺血管 / pulmonary vessel 混杂项使用 teacher-relative margin 0.01，并加入权重 0.05、温度 0.1 的 supervised contrastive loss。
- 对齐阶段冻结视觉编码器和语言模型，只训练共享对齐器；cosine 分类温度 0.07，另含 matching cosine attraction。
- 推理不读取 mask。病灶分数是“最佳病灶解剖相似度减最佳其他解剖相似度”；在最一致的 5x5 局部区域内选择 8 个 patch。
- patch 权重直接对病灶分数做 softmax；候选分数在各表型组内标准化后以 `tau=0.125` 做 softmax。
- 候选文本 token 序列通过重复末 token 补齐后加权求和，不做论文外的中位数范数重缩放。
- 融合 token 位于原生视觉 token 之后、任务指令之前；原生视觉 token 保留，decoder 完全冻结。

## 数据协议

配置记录的数据集 ID 为 `LIDC`，数据源根为 `/datasets/LIDC`。标签来自 LIDC-IDRI 放射科医师 XML 结节轮廓与表型评分、轮廓物理直径及 TotalSegmentator 2.18.0 解剖伪标签。患者级 `train/val/test` 划分统一绑定：

```text
/datasets/LIDC/pathology_encoder_r28_full_fov/splits.json
sha256: 3a22e64c084ba3327f3e73731a366385506a2e08e609d78c47f79182984de567
```

每次运行都会在读取配置时核对该哈希。新的数据 release 必须同时修改 `dataset.release`、源根、标签来源和 split manifest/hash，不能靠路径或实验 ID 猜测数据身份。

## 安装与运行

在服务器目录 `/autodl-tmp/wh/mebe/purevision` 中：

```bash
python -m pip install -e '.[preprocess,test]'
pytest
```

表型编码器两阶段训练：

```bash
torchrun --nproc_per_node=2 -m purevision.train \
  --config configs/01_phenotype_local.yaml

torchrun --nproc_per_node=2 -m purevision.train \
  --config configs/02_phenotype_global.yaml \
  --initialize-from runs/phenotype_local/best.pt
```

解剖编码器两阶段训练：

```bash
python scripts/train_anatomy_base.py --config configs/03_anatomy_base.yaml

python scripts/train_anatomy.py \
  --config configs/04_anatomy_final.yaml \
  --initialize-from runs/anatomy_base/best.pt
```

冻结特征提取和共享对齐：

```bash
python scripts/extract_alignment_features.py \
  --config configs/05_alignment.yaml --kind all

python scripts/train_alignment.py --config configs/05_alignment.yaml
```

R60 soft fusion 与解码：

```bash
python scripts/run_inference.py \
  --config configs/06_inference.yaml \
  --image /path/to/preprocessed_896x896.png \
  --instruction-file /path/to/task_instruction.txt \
  --output runs/inference/sample_result_zh.json
```

绑定历史冻结 checkpoint 的 MedGemma 1.5 复现 smoke 配置为
`configs/07_medgemma15_reproduction_smoke.yaml`。同一测试样本应分别运行标准未修改推理
baseline 与 PureVision 主方法，二者不能混写为同一条 pipeline：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_native_baseline.py \
  --config configs/07_medgemma15_reproduction_smoke.yaml \
  --image /path/to/test.png --sample-id SAMPLE_ID --split test \
  --instruction-file /path/to/task_instruction.txt \
  --output runs/medgemma15_reproduction_smoke/native_result_zh.json

CUDA_VISIBLE_DEVICES=1 python scripts/run_inference.py \
  --config configs/07_medgemma15_reproduction_smoke.yaml \
  --image /path/to/test.png --sample-id SAMPLE_ID --split test \
  --instruction-file /path/to/task_instruction.txt \
  --output runs/medgemma15_reproduction_smoke/purevision_result_zh.json
```

该配置在加载前核对 backbone 配置、权重索引和 R30/R39/R43 checkpoint 的
SHA-256。`smoke_only: true` 只证明真实权重端到端路径可运行，不代表已重跑论文的完整
VQA/RRG benchmark 或复现表中准确率。

## 特征与评估边界

原始预训练权重只用于分别初始化两个视觉 tower 以及冻结 decoder。标准未修改推理指原 backbone 自带视觉 tower 与 decoder；本方法的训练期表型 pooling 是 mask-conditioned pooling，但 mask 从不进入推理；本主链不使用 supervised centroid evaluation。共享对齐器对固定文本目标执行 cosine 匹配，属于冻结文本目标上的 zero-shot semantic classification，不等同于 hard centroid。

本目录不生成可视化。若另行绘制 t-SNE，只能将其描述为二维可视化，不能把图上的簇间空隙报告为定量 embedding 距离；定量关系必须在原始归一化表示空间中另算。

## 真实病例与报告解析

固定测试病例 `LIDC-IDRI-0079_s3_n0` 的图像、病灶 mask、来源和中英双语标签见 `examples/lidc_case_0079/`。先运行 `PYTHONPATH=src python scripts/run_real_case.py --check-only` 核验图像哈希、mask 和 4x4 位置；备齐 `07` 配置所列数据与 checkpoint 后运行 `PYTHONPATH=src python scripts/run_real_case.py` 完成原生 MedGemma 1.5 与 PureVision 配对生成。该单病例不构成论文 VQA/RRG 全量评估。

`scripts/parse_rrg_report.py` 使用 GPT6-Astra 解析报告中的 4x4 位置和七组表型；在本机设置 `OPENAI_API_KEY`，无需将密钥写入仓库。具体命令、解析范围和原始结果见 [README.md](README.md)。
