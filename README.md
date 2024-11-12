# MMDetection 3.0 Vector Extension English Guide
## Introduction
* MMDetection 3.0 Vector Extension is a deep learning framework developed based on [MMDetection](https://github.com/open-mmlab/mmdetection) for vector detection. For each ROI point in the image, we aim to have a corresponding vector and output the associated results.
* This project has been migrated and updated from [BONAI](https://github.com/jwwangchn/BONAI/), with detailed comments on important core code. Currently, the comments are in Chinese, but will be updated to English in the future.
## How to Use?
* This project now supports running inference locally!
* The test model is designed for predicting building footprint deviations based on satellite imagery.
* Execute the following command in the terminal (please modify the file path for the Inference script in advance):
```shell
python vec_demo/inferencer/inferenceBonai.py
```
* If you want to train/test this model locally, you can choose to use one of the two models in vec_demo/configs/ for training. Please change the paths for the training images and annotations before training.
```shell
python tools/train.py vec_demo/configs/loft_r50_fpn_25x_bonai.py
python tools/test.py vec_demo/configs/loft_r50_fpn_25x_bonai.py """(checkpoint location)""" --show
```
* Additionally, please note that the complete documentation for this project has not yet been written. Most of the code has been migrated from MMDetection, so please read the documentation of MMDetection first.
* You can check the specific core code location directly in the commit history, which includes information on changes to the core codebase. You can review the specific logic of the model and see which components have been added.