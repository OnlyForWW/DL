# 学习笔记
W创建和维护。
该仓库主要用于记录学习所得，内容为一家之言，仅供参考！

***
# 更新日志

- [2022-12-21](#2022-12-21)
- [2022-12-30](#2022-12-30)
- [2023-01-08](#2023-01-08)
- [2023-01-13](#2023-01-13)
- [2023-01-18](#2023-01-18)
- [2023-01-21](#2023-01-21)
- [2023-01-28](#2023-01-28)
- [2023-01-29](#2023-01-29)
- [Long term updates](#Long-term-updates)

## 2022-12-21
由W添加classification\RepVGG，内容包含RepVGG论文笔记、模型代码等，详情如下：
- code
  - ~~best.pth：Inter Image Classification使用模型RepVGG\_A1最佳训练权重，on train 94%，on val 89%。~~
  - model\_visualization.py：RepVGG\_A1模型可视化。
  - my\_train.py：模型训练脚本。
  - prediction.py：推理脚本。
  - repvgg.py：网络模型。
  - se\_block.py：部分RepVGG网络模型需使用SE模块。
- RepVGG.pdf：RepVGG论文笔记。
- RepVGG.pptx：论文汇报PPT。
- repvgg\_A1\_deploy.onnx.png：推理阶段模型可视化图。
- repvgg\_A1\_train.onnx.png：训练阶段模型可视化图。
## 2022-12-30
由W添加classification\ConvNeXt，内容包含ConvNeXt论文笔记、模型代码等，详情如下：
- code
  - convnext.py：网络模型。
- ConvNeXt.pdf：论文笔记。
- ConvNeXt.pptx：论文汇报PPT。
## 2023-01-08
由W添加tricks\DataBlance&Augmentation，内容包含数据平衡与增强代码和方法介绍。详情如下：
- code
  - Deal.py：执行脚本。
  - ImageOperate.py：数据平衡和增强方法类，包含图像处理和方法定义。
- README.md：相关文档。
- 示例图片：文档中的图片。
## 2023-01-13
由W添加detection\YOLO series\YOLOv7\backbone，主要内容包含YOLOv7使用的backbone-ELAN的相关介绍。
- ELAN.pdf：论文笔记。
- ELAN.pptx：论文汇报PPT。

更新：添加了ConvNeXt模型的预训练权重，详情见本仓库下的ConvNeXt文件夹。

## 2023-01-18
由W添加tricks\TricksForImageClassification，主要介绍CNN在分类任务上的训练技巧，论文提及了多种方法具有参考价值，详情如下：
- 论文笔记。
- 论文汇报PPT。

## 2023-01-21

由W添加classification\轻量化网络\MobileNet系列\MobileNetV1，主要包含了MobileNetV1相关内容，详情如下：

- MobileNetV1.pdf：论文笔记。
- MobileNetV1.pptx：论文汇报PPT。
- mobilenetv1.py：使用PyTorch构建的MobileNetV1网络模型。

如何使用我们提供的代码，下面给出示例：

```python 
from mobilenet import *

# 提供了三种不同宽度MobileNetV1模型
net = MobileNetV1_100()

# 可以自定义网络宽度，以0.25为例
net = MobilNet(cfgs=cfgs, ratio=0.25, **kwargs)
```

## 2023-01-28

由W添加timm\，主要介绍了timm库的使用：

- README.md：《PyTorch 图像分类模型（timm）：实用指南》的Markdown文档。
- PyTorch 图像分类模型（timm）：实用指南.pdf：由Markdown文档导出的PDF文件。

## 2023-01-29

由W添加classification\轻量化网络\MobileNet系列\ShuffleNetV1，主要包含了ShuffleNetV1相关内容，详情如下：

- ShuffleNetV1.pdf：论文笔记。
- ShuffleNetV1.pptx：论文汇报PPT。
- shufflenetv1.py：基于PyTorch构建的ShuffleNetV1模型。

如何使用我们所提供的代码，下面给出示例：

```python
from shufflenetv1 import *

# 我们提供了网络宽度为0.5、1.0、1.5和2.0，以及分组数为3和8
net = ShuffleNet_050_g3()
```

注意：如果您想自定义网络宽度（例如0.75）或者增加新的分组（例如4），您可能需要修改代码，并在修改时注意各个卷积层的输入和输出通道数，这将导致某些通道数不符合缩放比例。

# Long-term-updates

由W添加Vision Transformer，目前已包含如下内容：

- ViT
- DeiT

