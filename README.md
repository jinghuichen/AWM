# AWM
 
This repository contains our implementation of [One-shot Neural Backdoor Erasing via Adversarial Weight Masking](https://arxiv.org/abs/2207.04497) (accepted by NeurIPS 2022). \[[openreview](https://openreview.net/forum?id=Yb3dRKY170h)\]


## Prerequisites
* PyTorch
* CUDA


## Usage Examples:

```ruby
python main.py --dataset cifar10 --batch-size 128 --sample 500 --attack trojan-sq
python main.py --dataset gtsrb --batch-size 32 --sample 43 --trigger-norm 100 --attack a2a --arch small_vgg
python main.py --dataset cifar10 --batch-size 32 --sample 10 --trigger-norm 100 --attack badnets --alpha 0.99
```

We have released some checkpoints: https://drive.google.com/drive/folders/150Egil8yrot3ppdROf39fXDvh8jYYDvl?usp=sharing
Please add the datasets under `/data` folder to use.

## Citation
Please check our paper for technical details and full results. If you find our paper useful, please cite:

```
@inproceedings{
chai2022oneshot,
title={One-shot Neural Backdoor Erasing via Adversarial Weight Masking},
author={Shuwen Chai and Jinghui Chen},
booktitle={Thirty-Sixth Conference on Neural Information Processing Systems},
year={2022}
}
```
