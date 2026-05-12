# Segmentation Practice

这是一个用于医学图像语义分割微调实验的练习项目。当前项目围绕一个轻量级的 SegFormer 分割流程展开，目标是把一个原本偏 macOS/MPS 的 notebook 改造成可以在 Linux 服务器上稳定运行、可比较、可迭代的实验环境。

## 项目目标

- 在 Linux 环境下稳定运行分割训练与评估流程
- 统一训练和验证指标，方便比较不同微调策略
- 通过 `last/new` 双 notebook 的方式做版本迭代
- 用独立终端脚本复现实验结果，减少 notebook 评估不稳定带来的干扰

## 当前目录结构

```text
segmentation_practice/
├── README.md
├── __init__.py
├── evaluate_segmentation_versions.py
├── segmentation_improvements.py
├── segmentation_last.ipynb
├── segmentation_new.ipynb
├── segformer-finetuned-ade-improved/
├── segformer-finetuned-ade-short/
└── .gitignore
```

## 核心文件说明

### `segmentation_last.ipynb`

当前稳定基线版本。后续迭代中，如果 `new` 版本的实验结果更好，就会把更优实现回写到这里，作为下一轮的基线。

### `segmentation_new.ipynb`

下一轮改版草案。新策略优先写到这里，跑完后与 `last` 做同口径比较。

### `segmentation_improvements.py`

把 notebook 里的关键改动抽成可复用脚本，包含：

- 合成数据集构造
- 数据增强 `RandomAug`
- 自定义 `CustomTrainer`
- Dice + CrossEntropy 复合损失
- 冻结 encoder 的可选能力
- 适配 Linux 的训练入口

### `evaluate_segmentation_versions.py`

独立的终端评估脚本，用来公平比较 `last` 和 `new` 两个版本。它会输出：

- `training_loss`
- `pixel_accuracy`
- `mean_iou`

### `README.md`

本文件，说明项目背景、使用方式和当前迭代流程。

## 环境要求

项目默认使用 `med_phys` Conda 环境。已验证可用的关键依赖包括：

- Python 3.11
- PyTorch CPU 版
- `transformers`
- `datasets`
- `accelerate`
- `torchvision`
- `matplotlib`
- `Pillow`
- `numpy`

如果你已经创建了环境，可以直接激活：

```bash
conda activate med_phys
```

## 快速运行

### 1. 运行 notebook

在 VS Code 中打开 `segmentation_last.ipynb` 或 `segmentation_new.ipynb`，按顺序执行单元即可。

### 2. 运行改进脚本的 smoke test

```bash
/home/zhouyang/miniconda3/envs/med_phys/bin/python /home/zhouyang/segmentation_practice/segmentation_improvements.py --quick
```

### 3. 运行 last/new 终端对比脚本

```bash
/home/zhouyang/miniconda3/envs/med_phys/bin/python /home/zhouyang/segmentation_practice/evaluate_segmentation_versions.py --quick --seed 42
```

这个脚本会分别训练和评估 `last` 与 `new` 两个配置，并打印对比结果。

## 迭代流程

当前采用的约定是：

1. 先把新策略写到 `segmentation_new.ipynb`
2. 用同一套指标和同一份脚本跑 `last` 与 `new`
3. 如果 `new` 更好，就把它回写到 `segmentation_last.ipynb`
4. 下一轮再继续在 `segmentation_new.ipynb` 上尝试更进一步的改动

这样做的好处是，实验记录清晰，比较口径稳定，也便于回退。

## 当前实现要点

- 数据集使用本地合成图像，不依赖外部下载
- 标签处理保留 `255` 作为 ignore index
- 评估统一使用 `pixel_accuracy` 和 `mean_iou`
- 新版微调策略已经尝试过两阶段训练：先冻结 encoder，再解冻继续微调
- 所有实验都可以在终端独立复现，不只依赖 notebook

## 输出目录

脚本运行后会产生实验输出目录，例如：

- `segformer-finetuned-ade-short/`
- `segformer-finetuned-ade-improved/`

这些目录通常保存训练过程中的模型分片或中间结果，已在 `.gitignore` 中排除。

## 备注

- notebook 和脚本都已经迁移到 `/home/zhouyang/segmentation_practice`
- 后续所有版本迭代建议都遵循 `last/new` 双文件流程
- 如果你想进一步提升指标，下一步优先考虑更强的数据增强、学习率分段调度和更长训练周期
