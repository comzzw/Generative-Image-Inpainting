# Generative Image Inpainting with Segmentation Confusion Adversarial Training and Contrastive Learning
![figure](https://github.com/zzw-zjgsu/Generative-Image-Inpainting/blob/main/docs/framework.PNG?raw=true)
### [Arxiv Paper](https://arxiv.org/abs/2303.13133) | 

Generative Image Inpainting with Segmentation Confusion Adversarial Training and Contrastive Learning<br>
[Zhiwen Zuo](https://scholar.google.com/citations?user=ZDJKCGoAAAAJ&hl=en),  Lei Zhao, Ailin Li, Zhizhong Wang, Zhanjie Zhang, Jiafu Chen, Wei Xing, and Dongming Lu.<br>


<!-- ------------------------------------------------ -->
## Citation
If any part of our paper and code is helpful to your work, 
please generously cite and star us :kissing_heart: :kissing_heart: :kissing_heart: !

```
@article{zuo2023generative,
  title={Generative Image Inpainting with Segmentation Confusion Adversarial Training and Contrastive Learning},
  author={Zuo, Zhiwen and Zhao, Lei and Li, Ailin and Wang, Zhizhong and Zhang, Zhanjie and Chen, Jiafu and Xing, Wei and Lu, Dongming},
  journal={arXiv preprint arXiv:2303.13133},
  year={2023}
}
```


<!-- ---------------------------------------------------- -->
## Introduction 
We present a new adversarial training framework for image inpainting with segmentation confusion adversarial training (SCAT) and contrastive learning. 
1) **SCAT plays an adversarial game between an inpainting generator and a segmentation network, which provides pixel-level local training signals for our framework and can flexibly handle images with free-form holes.** 
2) **The proposed contrastive learning losses stabilize and improve our inpainting model's training by exploiting the feature representation space of the discriminator, in which the inpainting images are pulled closer to the ground truth images but pushed farther from the corrupted images.**

<!-- ------------------------------------------------ -->
## Results
![Places2_results](https://github.com/zzw-zjgsu/Generative-Image-Inpainting/blob/main/docs/places2_results.PNG?raw=true)

<!-- -------------------------------- -->
## Prerequisites 
* python 3.8.8
* [pytorch](https://pytorch.org/) (tested on Release 1.8.1)

<!-- --------------------------------- -->
## Installation 

Clone this repo.

```
git clone https://github.com/zzw-zjgsu/Generative-Image-Inpainting
cd Generative-Image-Inpainting/
```

For the full set of required Python packages, we suggest create a Conda environment from the provided YAML, e.g.

```
conda env create -f environment.yml 
conda activate inpainting
```

<!-- --------------------------------- -->
## Datasets 

1. download images and masks
2. specify the path to training data by `--dir_image` and `--dir_mask`.



<!-- -------------------------------------------------------- -->
## Getting Started

1. Training: 
    * Run 
    ```
    cd src 
    python train.py --dir_image [image path] --dir_mask [mask path] --dataset [Places2 or CelebA] --iterations [Places2:1e6 CelebA:3e5]
    ```
2. Resume training:
    ```
    cd src
    python train.py --resume 
    ```
3. Testing:
    ```
    cd src 
    python single_test.py --pre_train [path to pretrained model] --ipath [image path] --mpath [mask path] --outputs [out directory]
    ```
4. Evaluating (calucating PSNR/SSIM/L1):
    ```
    cd src 
    python test.py --pre_train [path to pretrained model] --dir_image [image path] --dir_mask [mask path] --outputs [out directory]
    ```

<!-- ------------------------------------------------------------------- -->
## Pretrained models (to be released)
[CELEBA] |
[Places2]

<!-- ------------------------ -->
## TensorBoard
Visualization on TensorBoard for training is supported. 

Run `tensorboard --logdir [log_folder] --bind_all` and open browser to view training progress. 



<!-- ------------------------ -->
## Acknowledgements

We would like to thank [aot-gan](https://github.com/researchmm/AOT-GAN-for-Inpainting) and [pytorch-msssim](https://github.com/VainF/pytorch-msssim). 

