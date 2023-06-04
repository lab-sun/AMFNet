# AMFNet-PyTorch

The official pytorch implementation of **Adaptive-Mask Fusion Network for Segmentation of Drivable Road and Negative Obstacle With Untrustworthy Features**.

Paper link: https://arxiv.org/abs/2304.13979

We test our code in Python 3.7, CUDA 11.1, cuDNN 8, and PyTorch 1.7.1. We provide `Dockerfile` to build the docker image we used. You can modify the `Dockerfile` as you want.  
<div align=center>
<img src="doc/network.png" width="900px"/>
</div>

# Demo

<!-- The accompanied video can be found at: https://www.youtube.com/watch?v=hQM5IW5N91M
<div align=center>
<a href="https://www.youtube.com/watch?v=hQM5IW5N91M"><img src="doc/demo_cover.PNG" width="70%" height="70%" />
</div> -->

# Introduction
AMFNet is a multi-modal fusion network for semantic segmentation of drivable road and negative obstacles.
# Dataset
We developed the [**NPO dataset**](https://ieeexplore.ieee.org/document/10114585/) to build our **DRNO dataset**. You can downloaded the **DRNO dataset** from [here](https://labsun-me.polyu.edu.hk/zfeng/AMFNet/). 
# Pretrained weights
The pretrained weight of AMFNet can be downloaded from [here](https://labsun-me.polyu.edu.hk/zfeng/AMFNet/).
# Usage
* Clone this repo
```
$ git clone https://github.com/lab-sun/AMFNet.git
```
* Build docker image
```
$ cd ~/AMFNet
$ docker build -t docker_image_amfnet .
```
* Download the dataset
```
$ (You should be in the AMFNet folder)
$ mkdir ./dataset
$ cd ./dataset
$ (download our preprocessed dataset.zip in this folder)
$ unzip -d .. dataset.zip
```
* To reproduce our results, you need to download our pretrained weights.
```
$ (You should be in the AMFNet folder)
$ mkdir ./weights_backup/AMFNet
$ cd ./weights_backup/AMFNet
$ (download our preprocessed dataset.zip in this folder)
$ unzip -d .. dataset.zip
$ docker run -it --shm-size 8G -p 1234:6006 --name docker_container_mafnet --gpus all -v ~/AMFNet:/workspace docker_image_amfnet
$ (currently, you should be in the docker)
$ cd /workspace
$ python3 run_demo.py
```
The results will be saved in the `./runs` folder.
* To train AMFNet
```
$ (You should be in the AMFNet folder)
$ docker run -it --shm-size 8G -p 1234:6006 --name docker_container_mafnet --gpus all -v ~/AMFNet:/workspace docker_image_amfnet
$ (currently, you should be in the docker)
$ cd /workspace
$ python3 train.py
```
* To see the training process
```
$ (fire up another terminal)
$ docker exec -it docker_container_amfnet /bin/bash
$ cd /workspace
$ tensorboard --bind_all --logdir=./runs/tensorboard_log/
$ (fire up your favorite browser with http://localhost:1234, you will see the tensorboard)
```
The results will be saved in the `./runs` folder.
Note: Please change the smoothing factor in the Tensorboard webpage to `0.999`, otherwise, you may not find the patterns from the noisy plots. If you have the error `docker: Error response from daemon: could not select device driver`, please first install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on your computer!

# Citation
If you use AMFNet in your academic work, please cite:
```
@ARTICLE{feng2023amfnet,
  author={Zhen Feng and Yuchao Feng and Yanning Guo and Yuxiang Sun},
  journal={IEEE Intelligent Vehicles Symposium}, 
  title={Adaptive-Mask Fusion Network for Segmentation of Drivable Road and Negative Obstacle With Untrustworthy Features}, 
  year={2023},
  volume={},
  number={},
  pages={},
  doi={}}
```

# Demo
<img src="doc/demo.png" width="700px"/>

# Acknowledgement
Some of the codes are borrowed from [RTFNet](https://github.com/yuxiangsun/RTFNet)

Contact: yx.sun@polyu.edu.hk

Website: https://yuxiangsun.github.io/
