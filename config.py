# -*- coding: utf-8 -*-
useInfoGan = True  # 是否使用infoGAN
useWasserstein = False  # 是否使用Wasserstein GAN
useImprovWasserstein = True  # 是否使用improved Wasserstein GAN
gradient_on_mean_varience = False  # 在判别器上BN是否在均值方差上求梯度
use_BN_dis = True  # 判别器是否使用BN
style_size = 32  # 噪声维数
num_category = 10  # 类别数
num_continuous = 0  # 设置2个连续变量
discriminator_lr = 1e-5  # 判别器的初始学习率
generator_lr = 1e-4  # 生成器的初始学习率
categorical_lambda = 1.0
continuous_lambda = 1.0
fix_std = True  # 把连续变量的方差固定为1.0
n_epochs = 30
batch_size = 256
plot_every = 200
ckpt = "ckpt"
weight_decay = 1e-6
clip_value = 0.01
disc_iter = 2
grad_lambda = 10
regularization = False





