# AWnet

A Dual Camera System for High Spatiotemporal Resolution Video Acquisition

[Project](https://NJUVISION.github.io/AWnet) **|** [Paper](https://arxiv.org/abs/1909.13051) **|** [video]()

Ming Cheng, [Zhan Ma](https://vision.nju.edu.cn/fc/d3/c29470a457939/page.htm), [M. Salman Asif](https://intra.ece.ucr.edu/~sasif/index.html), [Yiling Xu](http://english.seiee.sjtu.edu.cn/english/detail/2737_1313.htm), Haojie Liu, [Wenbo Bao](https://sites.google.com/view/wenbobao/home), and Jun Sun

IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)

![image](https://github.com/NJUVISION/AWnet/tree/master/images/image_system.jpg)

### Installation

The code has been tested with Python 3.7, PyTorch 1.0, CUDA 10.1 and Cudnn 7.6.4.

Once your environment is set up and activated, generate the Correlation package required by [PWCNet](https://github.com/NVlabs/PWC-Net/tree/master/PyTorch/external_packages/correlation-pytorch-master):

    $ cd correlation_package_pytorch1_0
    $ sh build.sh
    
### Demos
<!--哈哈我是注释，不会在浏览器中显示。
![image](https://github.com/NJUVISION/AWnet/blob/master/images/0.gif)
![image](https://github.com/NJUVISION/AWnet/blob/master/images/2.gif)
![image](https://github.com/NJUVISION/AWnet/blob/master/images/1.gif)
![image](https://github.com/NJUVISION/AWnet/blob/master/images/3.gif)
-->
#### Image demos
These images are captured with our dual iPhone 7 cameras.
<div align="center">
    <img src="https://github.com/NJUVISION/AWnet/blob/master/images/0.gif" height="250"/><img src="https://github.com/NJUVISION/AWnet/blob/master/images/2.gif" height="250"/><img src="https://github.com/NJUVISION/AWnet/blob/master/images/1.gif" height="250"/><img src="https://github.com/NJUVISION/AWnet/blob/master/images/3.gif" height="200"/>  
</div>

#### Video Demos
Different illumination conditions:
[High Light Illumination](http://yun.nju.edu.cn/f/5087da2041/?raw=1) **|** [Medium Light Illumination](http://yun.nju.edu.cn/f/c251103e67/?raw=1) **|** [Low Light Illumination](http://yun.nju.edu.cn/f/42a121985b/?raw=1)

Single-Reference vs Multi-Reference:
[Simulated data](http://yun.nju.edu.cn/f/f8c3604d6a/?raw=1) **|** [Real data](http://yun.nju.edu.cn/f/76f3cea6da/?raw=1)

#### [Pretrained models](http://yun.nju.edu.cn/d/b1d5b3c3a3/)
<!--[Model without noise](http://yun.nju.edu.cn/f/a5a4646864/?raw=1) **|** [Model with noise (0.01)](http://yun.nju.edu.cn/f/baf2ba5663/?raw=1)-->

