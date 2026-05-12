Segmentation improvements
=========================

包含用于对 `Segmentation_test.ipynb` 中训练流程进行小规模改进的独立脚本 `segmentation_improvements.py`。

快速用法（在 `med_phys` 环境中运行）:

```bash
# 进入 conda env（如果尚未激活）
conda activate med_phys

# 运行短时 smoke test (小数据集、1 epoch)
/home/zhouyang/miniconda3/envs/med_phys/bin/python medical_physics/segmentation_improvements.py --quick
```

脚本内容要点：
- 简单合成数据生成器（与 notebook 保持类似）
- `RandomAug`：镜像与 90 度旋转增强（同时应用于图像和 mask）
- `CustomTrainer`：重载 `compute_loss` 以使用 Dice + CrossEntropy 复合损失
- 支持 `fp16`（仅在 CUDA 可用时启用）和梯度累积参数

终端对比脚本：

```bash
/home/zhouyang/miniconda3/envs/med_phys/bin/python medical_physics/evaluate_segmentation_versions.py --quick
```

这个脚本会分别跑 `last` 和 `new` 两个配置，并打印 `training_loss`、`pixel_accuracy` 和 `mean_iou` 对比。

后续建议：
- 将此脚本中改动逐步合并回 notebook 的训练单元（替换 dataset / trainer 定义）
- 若有 GPU，可启用更丰富的增强、冻结 encoder 以及更长的训练周期
