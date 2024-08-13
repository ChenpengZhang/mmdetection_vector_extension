# MMDetection 3.0 Vector Extension 中文指南
## 简介
* MMdetection 3.0 Vector Extension 是基于 [MMDetection](https://github.com/open-mmlab/mmdetection) 开发的一套用于向量检测的深度学习框架。对于图中的每一个ROI点，我们希望有一个向量能够与之进行对应，并输出这个对应的结果。
* 本项目主要从[BONAI](https://github.com/jwwangchn/BONAI/)迁移更新而来，并对重要的核心代码做了详细的注释。目前注释还是中文版本，后续会改为英文。
## 如何使用？
* 本项目的运行还需要依赖于配置文件以及训练和测试数据集，因此目前还无法运行。后续会在其它的repo中上传这些数据。
* 另外请注意，目前本项目还没有编写完整的文档，大部分代码仍然是从MMDetection中迁移过来，所以请先阅读[MMDetection](https://github.com/open-mmlab/mmdetection)的文档。
* 具体核心代码位置可以直接查看[提交记录](https://github.com/ChenpengZhang/mmdetection_vector_extension/commit/bab4e85815ac08b98cb711a6ee25cb2cb571184b), 里面包含了核心的代码库改动信息，可以逐一查看模型的具体逻辑，以及什么组件被添加了进去。