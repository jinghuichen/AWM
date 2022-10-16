# AWM
 
This repository contains our implementation of [One-shot Neural Backdoor Erasing via Adversarial Weight Masking](https://arxiv.org/abs/2207.04497) (accepted by NeurIPS 2022). \[[openreview](https://openreview.net/forum?id=Yb3dRKY170h)\]


## Prerequisites
* PyTorch
* CUDA


## Usage Examples:

```ruby
python AWM.py --dataset gtsrb --batch-size 32 --sample 43 --trigger-norm 100 --attack a2a --arch small_vgg
```

We have released some checkpoints: https://drive.google.com/drive/folders/150Egil8yrot3ppdROf39fXDvh8jYYDvl?usp=sharing


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
