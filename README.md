# Deep Iterative Residual Convolutional Network for Single Image Super-Resolution (ISRResCNet)
![](https://img.shields.io/badge/pytorch-ISRResCNet-green)

An official PyTorch implementation of the [ISRResCNet](https://github.com/RaoUmer/ISRResCNet) network as described in the paper **[Deep Iterative Residual Convolutional Network for Single Image Super-Resolution](https://arxiv.org/abs/2009.04809)** which is published in the 25th International Conference of Pattern Recognition (ICPR), 2020.

✨ _**Visual examples**_:

[<img src="figs/vis_res1.PNG" width="350px"/>](https://imgsli.com/NDg4ODY) [<img src="figs/vis_res2.PNG" width="520px"/>](https://imgsli.com/NDg4ODc)

#### Abstract
> Deep convolutional neural networks (CNNs) have recently achieved great success for single image super-resolution (SISR) task due to their powerful feature representation capabilities. The most recent deep learning based SISR methods focus on  designing deeper / wider models to learn the non-linear mapping between low-resolution (LR) inputs and high-resolution (HR) outputs. These existing SR methods do not take into account the image observation (physical) model and thus require a large number of network's trainable parameters with a great volume of training data. To address these issues, we propose a deep Iterative Super-Resolution Residual Convolutional Network (ISRResCNet) that exploits the powerful image regularization and large-scale optimization techniques by training the deep network in an iterative manner with a residual learning approach. Extensive experimental results on various super-resolution benchmarks demonstrate that our method with a few trainable parameters improves the results for different scaling factors in comparison with the state-of-art methods.

#### Oral Presentation (Video)
[![Video](https://img.youtube.com/vi/4TLjeIYuOyQ/hqdefault.jpg)](https://youtu.be/4TLjeIYuOyQ)

#### BibTeX
    @InProceedings{Umer_2020_ICPR,
                   author = {Muhammad Umer, Rao and Luca Foresti, Gian and Micheloni, Christian},
                   title = {Deep Iterative Residual Convolutional Network for Single Image Super-Resolution},
                   booktitle = {Proceedings of the International Conference of Pattern Recognition (ICPR)},
                   month = {January},
                   year = {2021}
                  }

## Quick Test

This model can be run on arbitrary images with a Docker image hosted on Replicate: https://beta.replicate.ai/RaoUmer/ISRResCNet. Below are instructions for how to run the model without Docker:

#### Dependencies
- [Python 3.7](https://www.anaconda.com/distribution/) (version >= 3.0)
- [PyTorch >= 1.0](https://pytorch.org/) (CUDA version >= 8.0 if installing with CUDA.)
- Python packages:  `pip install numpy opencv-python`

#### Test models
1. Clone this github repository as the following commands: 
```
git clone https://github.com/RaoUmer/ISRResCNet
cd ISRResCNet
cd isrrescnet_code_demo
```
2. Place your own **low-resolution images** in the `./isrrescnet_code_demo/LR` folder. (There are two sample images i.e. set5_img_butterfly_x4 and urban100_img_092_x4). 
3. Run the test by the provided script `test_isrrescnet.py`.
```
python test_isrrescnet.py       
```
4. The SR results are in the `./isrrescnet_code_demo/sr_results` folder.

## ISRResCNet Architecture
#### Overall Representative diagram
<p align="center">
  <img height="250" src="figs/isrrescnet.gif">
</p>

#### ERD block
<p align="center">
  <img height="150" src="figs/rescnet.png">
</p>

## Quantitative Results
Average PSNR/SSIM values for scale factors x2, x3, and x4 with the bicubic degradation model. The best performance is shown in **red** and the second best
performance is shown in **blue**.
<p align="center">
  <img height="200" src="figs/quant_res.PNG">
</p>

## Visual Results
Visual comparison of our method with other state-of-the-art methods on the x4 super-resolution over the SR benchmarks. For visual comparison on the benchmarks, you can download our results from the Google Drive: [ISRResCNet](https://drive.google.com/drive/folders/1IioErwfd1cjfBMBOjUzH1guWuI-iZzFm?usp=sharing).
<p align="center">
  <img height="250" src="figs/res1.png">
</p>

<p align="center">
  <img height="250" src="figs/res2.png">
</p>

## Code Acknowledgement
The training codes is based on [burst-photography](https://github.com/cig-skoltech/burst-cvpr-2019) and [deep_demosaick](https://github.com/cig-skoltech/deep_demosaick).
