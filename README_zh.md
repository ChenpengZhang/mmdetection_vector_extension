# MMDetection 3.0 Vector Extension 中文指南
## 简介
* MMdetection 3.0 Vector Extension 是基于 [MMDetection](https://github.com/open-mmlab/mmdetection) 开发的一套用于向量检测的深度学习框架。对于图中的每一个ROI点，我们希望有一个向量能够与之进行对应，并输出这个对应的结果。
* 本项目主要从[BONAI](https://github.com/jwwangchn/BONAI/)迁移更新而来，并对重要的核心代码做了详细的注释。目前注释还是中文版本，后续会改为英文。
## 如何使用？
* 现在本项目已经支持在本地运行Inference！
* 测试模型为基于BONAI的对卫星影像而言的建筑脚印偏离预测。
* 在命令行中执行（执行之前请先修改脚本内部的Inference文件路径为自己的）：
```shell
python vec_demo/inferencer/bonai_inferencer.py
```
* 如果你想要在本地训练/测试这个模型，则可以选择使用vec_demo/configs/中的两个模型进行训练，在训练前也请将训练图片和标注的路径改为自己的。
```shell
python tools/train.py vec_demo/configs/loft_r50_fpn_25x_bonai.py
python tools/test.py vec_demo/configs/loft_r50_fpn_25x_bonai.py """(checkpoint location)""" --show
```
* 另外请注意，目前本项目还没有编写完整的文档，大部分代码仍然是从MMDetection中迁移过来，所以请先阅读[MMDetection](https://github.com/open-mmlab/mmdetection)的文档。
* 具体核心代码位置可以直接查看[提交记录](https://github.com/ChenpengZhang/mmdetection_vector_extension/commit/bab4e85815ac08b98cb711a6ee25cb2cb571184b), 里面包含了核心的代码库改动信息，可以逐一查看模型的具体逻辑，以及什么组件被添加了进去。