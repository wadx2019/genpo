# [NeurIPS 2025] GenPO: Generative Diffusion Models Meet On-Policy Reinforcement Learning

Code release for **GenPO: Generative Diffusion Models Meet On-Policy Reinforcement Learning (NeurIPS 2025)**.

[[paper]](https://arxiv.org/abs/2505.18763) [[project page]](https://dingsht.tech/genpo-webpage/)

![](./asset/genpo.png)

## Requirements
OS version: Ubuntu 22.04
A suitable [conda](https://conda.io) environment named `genpo` can be created and activated with:
```
conda create -n genpo python=3.10
conda activate genpo
```
To get started, install IsaacSim=4.5 in [IsaacSim](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/installation/pip_installation.html).
```
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
```
install our customized isaaclab in the current directory using
```
./isaaclab.sh --install # or "./isaaclab.sh -i"
```
install flow_rsl_rl with
```
./isaaclab.sh -i rsl_rl
./isaaclab.sh -p -m pip install -e flow_rsl_rl/
```
## Running
Running experiments based on our code could be quite easy, so below we use `Isaac-Ant-v0` task as an example. 

```
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/trainflow.py --task=Isaac-Ant-v0 --num_envs 1024 --headless
```

## Citation
If you find this repository useful in your research, please consider citing:

```
@inproceedings{
ding2025genpo,
title={Gen{PO}: Generative Diffusion Models Meet On-Policy Reinforcement Learning},
author={Shutong Ding and Ke Hu and Shan Zhong and Haoyang Luo and Weinan Zhang and Jingya Wang and Jun Wang and Ye Shi},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=BmRNz1TpCc}
}
```

## Acknowledgement

The code of GenPO is based on the implementation of [IsaacLab](https://github.com/BellmanTimeHut/DIPO).
