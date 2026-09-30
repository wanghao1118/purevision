# PureVision 数据构造与论文一致性核查

核查说明版本：2026-09-30-v2。本次仅修订评测口径与说明，不修改既有 checkpoint、权重、原始嵌入或历史审计记录。

本仓库只发布代码、构造配方和核查方法，不发布 CBIS-DDSM、LIDC-IDRI、3DReasonKnee 的原始或处理后影像、逐例清单、掩码、权重、原始嵌入或历史审计资产。新构造结果须保存在仓库外，并以 `dataset_contract.json` 随数据集交付。共享推理、报告解析和评测从该规范读取解剖与表型维度；数据集 ID 和类别 ID 保持稳定，中英显示名另存。已有 LIDC 训练脚本仍与 LIDC 专有评分空间绑定，不能把这三套数据直接替换路径后宣称完成联合训练。

## 固定来源

| 数据集与发布版 | 原始来源根目录 | 标签来源 | 患者级 train/val/test 清单及 SHA-256 | 冻结 test 与可定位数 |
|---|---|---|---|---|
| LIDC-IDRI 肺结节 (pulmonary nodule)，`LIDC`，`lidc_purevision_r60_20260830_v1` | `/datasets/LIDC` | 医师 XML 轮廓、原始评分及物理直径；TotalSegmentator 2.18.0 解剖伪标签 | `/datasets/LIDC/pathology_encoder_r28_full_fov/splits.json`，`3a22e64c084ba3327f3e73731a366385506a2e08e609d78c47f79182984de567` | 501；单格病灶 425 |
| CBIS-DDSM 乳腺病灶 (breast lesion)，`CBIS-DDSM`，`cbis_native_four_part_v2_20260918` | `/mnt/sda/hao/wh/datasets/CBIS-DDSM` | DICOM ROI；GrabCut 乳腺组织 (breast tissue)；Attention U-Net 胸肌 (pectoral muscle) 伪标签；CSV 肿块形态 (mass shape)、肿块边缘 (mass margins)、钙化分布 (calcification distribution) | `/datasets/cbis_ddsm/image_lesion_anatomy_text_native_v2_20260918/manifest.jsonl`，`81991e04e7a5cfc50f54614c66e2e30417c3ab2985e2813b527d42c61b5f9b62` | 540；单格 ROI 227 |
| 3DReasonKnee 内侧半月板 (medial meniscus)，`3DReasonKnee`，`medial_meniscus_strict2d_crop128_v4` | `/datasets/3dreasonknee` | 检查级区域 MOAKS 内侧半月板外突等级 (medial meniscus medial extrusion grade)；人工优先、模型补全的解剖 bitset；整块内侧半月板代理 ROI 用作本评测定位参考 | `/datasets/3dreasonknee/medial_meniscus_strict2d_crop128_v4/manifest/all.jsonl`，`6fe7bc1aad69e79fa3103ba0ffaa5d832bfac2309b629fe701d4bc73343b8d51` | 1203；可定位参考 1198 |

以上哈希来自 2026-09-30 对服务器冻结文件的读取，非构造脚本对新输出的预设结果。路径只是来源记录，不作为数据集身份的唯一依据。每次重构都必须保存新 release、新哈希和中文记录，不能覆盖历史 checkpoint、权重、原始特征或审计文件。

## 构造顺序

在有数据和依赖的服务器上执行，所有输出目录都应位于代码仓库外。`PYTHONPATH=src` 用于共享模块；构造脚本本身不下载影像。

**LIDC-IDRI 肺结节 (pulmonary nodule)。** `purevision.preprocess` 从原始 DICOM 与 pylidc 医师 XML 生成裁剪中间产物，保留六项原始医师共识评分；`purevision.splits` 生成患者级划分；`build_lidc_full_fov.py` 再按来源裁剪坐标逐像素核验全幅 DICOM，恢复全幅图和结节 (nodule) mask，不使用肺裁剪作为最终图像。

```bash
PYTHONPATH=src python -m purevision.preprocess --config configs/01_phenotype_local.yaml --output-dir /datasets/LIDC/new_lung_crop_intermediate
PYTHONPATH=src python -m purevision.splits --config configs/01_phenotype_local.yaml --processed-dir /datasets/LIDC/new_lung_crop_intermediate
PYTHONPATH=src python scripts/build_lidc_full_fov.py --source /datasets/LIDC/new_lung_crop_intermediate --output /datasets/LIDC/new_full_fov_release
PYTHONPATH=src python scripts/build_dataset_contract.py --recipe dataset_recipes/lidc.yaml --dataset-root /datasets/LIDC/new_full_fov_release
```

`build_lidc_full_fov.py` 遇到旧裁剪图与原始 DICOM 像素不一致会失败，不会默默生成不同病例。新划分哈希若与历史 R28 不同，必须作为新 release 处理，不能直接套用历史 R30/R39/R43 checkpoint 的比较结论。解剖伪标签阶段另需 TotalSegmentator 权重与原始解剖训练构造流程；这里只提供病灶表型全幅图重建及冻结对齐词表配方，不宣称已从原始 DICOM 独立重建 R33/R34 解剖训练集。

冻结 R43 对齐器的文本目标可以独立检查：`PYTHONPATH=src python scripts/check_checkpoint_contract.py --dataset-contract /datasets/LIDC/pathology_encoder_r28_full_fov/dataset_contract.json --alignment-checkpoint /models/selected_r43_shared_alignment.pt`。2026-09-30 在 Pro60002 上使用历史 R43 checkpoint 核对通过，九个解剖候选及七组表型候选的 ID 和顺序一致；这只证明词表契约一致，不代表按本次修正的损失重新训练。

**CBIS-DDSM 乳腺病灶 (breast lesion)。** [`dataset_construction/build_cbis_native_four_part_dataset.py`](dataset_construction/build_cbis_native_four_part_dataset.py) 从原始 DICOM、原有 v1 样本 ID/标签/划分及既有胸肌修订产物，唯一匹配全图与 ROI；使用原生分辨率图、种子化 GrabCut 乳腺组织 (breast tissue) 和冻结 Attention U-Net 胸肌 (pectoral muscle) 伪标签。每例要求来源 ROI、几何与哈希匹配，病灶 ROI 位于乳腺前景比例至少 97.5%；失败例保留排除原因。`--v1-root` 和 `--revision-root` 是重建所需的历史中间产物，不在仓库中；原始胸肌 checkpoint 亦需在本地单独准备。

```bash
python dataset_construction/build_cbis_native_four_part_dataset.py --source-root /mnt/sda/hao/wh/datasets/CBIS-DDSM --v1-root /datasets/cbis_ddsm/image_lesion_anatomy_text_v1 --revision-root /datasets/cbis_ddsm/native_revision_v1 --output /datasets/cbis_ddsm/new_native_release
PYTHONPATH=src python scripts/build_dataset_contract.py --recipe dataset_recipes/cbis.yaml --dataset-root /datasets/cbis_ddsm/new_native_release
```

这里的 `v1`、`native_revision_v1` 路径只是参数示例，须替换为实际冻结产物；若缺失，不得以相近 ROI 或新模型输出假冒相同 release。CBIS 多值标签保留完整组合，由构造后数据的候选词表决定共享对齐和解析维度，不在共享算法内写死。

**3DReasonKnee 内侧半月板 (medial meniscus)。** [`dataset_construction/build_knee_native_subset.py`](dataset_construction/build_knee_native_subset.py) 从 3DReasonKnee/OAI 来源建立患者级子集，接着 [`dataset_construction/prepare_knee_strict2d.py`](dataset_construction/prepare_knee_strict2d.py) 导出单张近冠状位原生 MRI 和人工掩码，最后 [`dataset_construction/build_knee_strict2d_dataset.py`](dataset_construction/build_knee_strict2d_dataset.py) 以冻结二维模型补全缺失结构，保留八个独立 bitset 通道和原生整数裁剪。`prepare`、`build`、`verify` 子命令分别准备、生成、核验；必须提供源 MOAKS 清单、已核查的 pilot 与冻结模型权重，不能只拿输出目录重建来源。

```bash
python dataset_construction/build_knee_strict2d_dataset.py prepare --input /datasets/3dreasonknee/strict2d_source --output /datasets/3dreasonknee/new_v4 --model /models/coronal_best_model.h5 --pilot /audits/knee_45_case_pilot
python dataset_construction/build_knee_strict2d_dataset.py build --output /datasets/3dreasonknee/new_v4
python dataset_construction/build_knee_strict2d_dataset.py verify --output /datasets/3dreasonknee/new_v4
PYTHONPATH=src python scripts/build_dataset_contract.py --recipe dataset_recipes/knee.yaml --dataset-root /datasets/3dreasonknee/new_v4
```

膝关节解剖包含股骨 (femur)、胫骨 (tibia)、髌骨 (patella)、软骨 (cartilage) 和半月板 (meniscus) 的可重叠通道。本评测约定将整块内侧半月板 (medial meniscus) 的 `lesion_roi_proxy` 视为定位参考真值；跨格时以阳性像素最多的网格为答案，并列按行列顺序取首格。它不是专家逐像素标注的外突病灶 (extrusion lesion) 区域，必须保留这一来源说明。

## 统一核查与评测

```bash
PYTHONPATH=src python scripts/validate_dataset_contract.py --dataset-contract /datasets/<release>/dataset_contract.json --output /audits/<release>_核查_ZH.json
PYTHONPATH=src python scripts/build_benchmark.py --dataset-contract /datasets/<release>/dataset_contract.json --output-dir /benchmarks/<release> --phenotype-questions 200 --grounding-questions 200 --rrg-cases 200
```

膝关节使用 `--phenotype-questions 100 --grounding-questions 100 --rrg-cases 183`。2026-09-30 在 Pro60002 的冻结 v4 test 上实测：1203 例中 1198 例有可定位代理 ROI，成功生成 100 道定位题、100 道表型题和 183 个 RRG 参考病例。LIDC 使用上面的 200/200/200 参数，实测七个表型维度各 200 道；二类、三类维度按原生候选数出题，评测者也可提供自己的题目清单，评分入口按清单执行。论文原始四选一干扰项、题号和 GPT6-Astra 历史提示词未发布，因此这些新题不冒充论文原题。`reference.jsonl` 是 test 全集，RRG 评分须使用 `rrg_reference.jsonl`。生成的参考清单、题目和掩码都属于数据，不得加入 Git。评分用 [`scripts/score_benchmark.py`](scripts/score_benchmark.py)；GPT6-Astra 报告解析用 `scripts/parse_rrg_report.py --dataset-contract ...`，只需用户在自己的环境中设置 `OPENAI_API_KEY`。所有缺失、无效、冲突的字段按错误处理。

## 论文统计与数据口径

论文中 CBIS-DDSM 的 2543 和 3DReasonKnee 的 7846 是全量样本数，写作时误标为 test；冻结患者级 test 分别为 540 和 1203。这是论文文字勘误，不是代码划分错误，不能通过重命名清单修改。论文的 12120 张 LIDC 解剖切片与冻结全幅病灶表型集的 2634 个病灶样本属于不同统计单元，也不是代码与论文的实现差异。

膝关节论文描述五类解剖及主要来自数据集提供的 nnU-Net 掩码；当前 v4 来源记录含八个可重叠通道，缺失人工通道由另一冻结二维模型补全。这是数据构造来源和类别粒度的记录，不能仅凭数量判断共享算法写死或出错，也不能把两套掩码来源说成完全相同。实际类别及表型维度始终由构造后数据规范提供。

## 代码实现范围与技术说明

历史 R30 checkpoint 形成于本仓库关系损失公式修正前；修正代码不等于已用新公式重训。当前 LIDC 完整训练入口仍绑定七维目标空间，CBIS-DDSM 和 3DReasonKnee 的完整三阶段训练尚未统一实现；已统一的是数据规范、推理候选读取、报告解析、评测构造和评分。仓库不上传 checkpoint。

训练图像路径采用 BICUBIC 插值，推理路径调用原生 MedGemma processor，二者存在预处理实现差异。这是上传代码内部的训练/推理核验事项，不是已证实的“代码与论文不一致”：论文未给出足以判定像素级一致性的插值细节。历史权重核验时应按其原训练路径记录预处理，不得静默切换后宣称精确复现。

原始预训练权重的标准未修改推理、训练时 mask 条件池化、监督质心评估与零样本分类是不同实验，不可混称。任何 t-SNE 图仅用于二维可视化；定量距离必须在原始归一化嵌入空间另算。文中数据数量来自服务器冻结清单核查，并非论文结果表格的重算准确率。
