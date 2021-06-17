# Art generation setup for Imaginative Walks paper.

Note: This repository is based on [rosinality/stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch)  
It contains the function implementation of the GRaWD loss proposed in Imaginative walks paper. It contains the integration of loss with only StyleGAN2.

## Requirements

I have tested on:

* PyTorch 1.3.1
* CUDA 10.1/10.2

## Usage

```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=12895 train.py\
        --batch=32\
        --checkpoint_folder=styleGAN2\
        --n_sample=25\
        --size=256\
        --name_suffix=RW-W10-T10\
        --use_RW\
        --normalize_protos_scale=3.0\ # normalization scale
        --RW_weight=10.0\ # GRaWD weight
        --RW_tau=3\ # Number of steps
        --no_pbar\ 
```
