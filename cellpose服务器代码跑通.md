# 相关问题
1. 要把代码在服务器上跑通，弄清楚输入输出
2. 输出的图片位置和像素是否与输入一致，输出的分割标注能否按照根据原图的拼接规则进行拼接？
3. 能否输出一个只有像素点分割的标注的文件？
4. 输入除了tif以外的那个npy文件里面的具体内容和格式？
5. 输出的分割文件格式是怎么样的？
6. 需要多少张图片进行fine-tuning？
7. 如果我们要手动分割喂给神经网络训练，需要怎么标注？

## 1.关于代码问题
1.输入为一张1G多的整图，现有问题如下
内存开销比较大（峰值远超400G，主要是在读取npy文件时内存溢出），不能在jupyter内展示图片实在过大加载太慢，最后爆内存本次测试失败。目前的方法只有将模型训练好进行一张一张图跑。
2.在跑体积比较大的图的时候需要对python的内核相关参数进行修改，不然读不进图
```
vim ~/miniconda3/envs/cellpose/lib/python3.9/site-packages/numpy/lib/format.py
```
需要将miniconda里cellpose环境中的python的format.py文件相关参数修改
```
pickle.dump(array, fp, protocol=3, **pickle_kwargs) -> pickle.dump(array, fp, protocol=4, **pickle_kwargs)
```
需要把protocol版本改为4
3.个人建议是在本地用GUI将相关模型训练好，然后将训练好的模型在服务器上跑一张整图而不是分割为若干份的小图（免去了做完细胞分割之后再将图像拼接为一张整图的步骤）。跑出来的图形与原图形的像素能对应上，输出的图片位置和像素与输入一致。
![输入图片说明](https://s2.loli.net/2023/09/25/9oajSuqtfyVrgAk.jpg)
## 2.代码
```
%%time
from cellpose import utils, io, models
from cellpose import plot

import matplotlib.pyplot as plt

# 读取需要输入的图像文件
filename = 'stitch_DAPI.tiff'
img = io.imread(filename)

# 设置训练集（nuclei为预设训练集）
# model = models.Cellpose(gpu=True, model_type='nuclei')

# 设置训练集为自己的训练集，model_path为模型路径
model_path = "test_1"
model = models.CellposeModel(gpu=True,pretrained_model=model_path)

# diameter of cells (set to zero to use diameter from training set):
diameter =  0
diameter = model.diam_labels if diameter==0 else diameter

# define CHANNELS to run segementation on
# grayscale=0, R=1, G=2, B=3
# channels = [cytoplasm, nucleus]
chan = [0,0]

# 进行图像分割
masks, flows, styles = model.eval(img, diameter=diameter, channels=chan)

# 输出格式为seg的文件（该文件适用于在cellpose的gui中打开以查看具体分割情况，同时也能作为训练集输入）
io.masks_flows_to_seg(img, masks, flows, diameter, filename, chan)

# 保存结果
io.save_to_png(img, masks, flows, filename)

# 在jupyter中展示分割效果(如果图像过大建议不要展示，例如跑的是1G的原图）
# fig = plt.figure(figsize=(12,5))
# plot.show_segmentation(fig, img, masks, flows[0], channels=chan)
```
运行代码后会输出如下的相关文件![cellpose输出](https://s2.loli.net/2023/09/25/i74AnEbFeC6u5fO.jpg)
test.jpg为输入文件
test_cp_masks.png为分割的细胞（由于channel选择是0，0所以这个预览图看起来像全黑）
test_cp_outlines.txt为分割的细胞边界坐标信息，每一行为一个细胞，详细如下
![细胞边界坐标信息](https://s2.loli.net/2023/09/25/EQk7o8PUFxTKJWw.jpg)
test_cp_output.png为综合输出
![输入图片说明](https://s2.loli.net/2023/09/25/Q3aOAqlUGTEhryJ.jpg)
seg文件为一个综合性的文件，主要作用是作为训练文件，可以在cellpose的GUI中打开进行手动标注细胞边界
## seg文件中相关内容
使用如下代码查看seg文件内容
```
import numpy as np
from cellpose import plot, utils
dat = np.load('test_seg.npy', allow_pickle=True).item()

dat
```
输出如下
```
{'img': array([[[44, 44, 44],
         [46, 46, 46],
         [46, 46, 46],
         ...,
         [49, 49, 49],
         [45, 45, 45],
         [49, 49, 49]],
 
        [[37, 37, 37],
         [37, 37, 37],
         [34, 34, 34],
         ...,
         [44, 44, 44],
         [48, 48, 48],
         [45, 45, 45]],
 
        [[32, 32, 32],
         [32, 32, 32],
         [32, 32, 32],
         ...,
         [50, 50, 50],
         [47, 47, 47],
         [45, 45, 45]],
 
        ...,
 
        [[31, 31, 31],
         [32, 32, 32],
         [33, 33, 33],
         ...,
         [34, 34, 34],
         [37, 37, 37],
         [35, 35, 35]],
 
        [[35, 35, 35],
         [37, 37, 37],
         [39, 39, 39],
         ...,
         [31, 31, 31],
         [34, 34, 34],
         [34, 34, 34]],
 
        [[42, 42, 42],
         [42, 42, 42],
         [45, 45, 45],
         ...,
         [38, 38, 38],
         [34, 34, 34],
         [38, 38, 38]]], dtype=uint8),
 'outlines': array([[0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        ...,
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0]], dtype=uint16),
 'masks': array([[0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        ...,
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0]], dtype=uint16),
 'chan_choose': [0, 0],
 'ismanual': array([False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False]),
 'filename': 'test.jpg',
 'flows': [array([[[[ 5, 28,  9],
           [ 3, 32, 15],
           [ 3, 30, 12],
           ...,
           [ 3, 34, 57],
           [ 4, 25, 52],
           [ 7, 15, 46]],
  
          [[13, 19,  0],
           [ 6,  9,  0],
           [ 5, 10,  1],
           ...,
           [10, 21, 62],
           [10, 16, 52],
           [19,  8, 55]],
  
          [[ 5,  3,  0],
           [ 2,  3,  0],
           [ 2,  3,  0],
           ...,
           [21, 11, 64],
           [22,  6, 53],
           [25,  5, 55]],
  
          ...,
  
          [[ 1,  8,  3],
           [ 0,  3,  0],
           [ 1,  6,  1],
           ...,
           [ 0,  0,  1],
           [ 2,  0,  5],
           [ 9,  1, 17]],
  
          [[ 9, 20,  2],
           [ 7, 12,  0],
           [ 9, 13,  0],
           ...,
           [ 6,  3,  0],
           [18,  1,  8],
           [17,  0, 20]],
  
          [[25, 25,  0],
           [27, 22,  0],
           [28, 24,  0],
           ...,
           [29,  8,  6],
           [31,  5, 10],
           [31,  0, 22]]]], dtype=uint8),
  array([[[147, 152, 150, ..., 160, 160, 153],
          [144, 140, 132, ..., 162, 158, 156],
          [131, 124, 128, ..., 166, 161, 158],
          ...,
          [130, 131, 134, ..., 138, 145, 147],
          [143, 143, 144, ..., 137, 144, 150],
          [147, 152, 152, ..., 149, 149, 154]]], dtype=uint8),
  array([[[[0, 0, 0],
           [0, 0, 0],
           [0, 0, 0],
           ...,
           [0, 0, 0],
           [0, 0, 0],
           [0, 0, 0]],
  
          [[0, 0, 0],
           [0, 0, 0],
           [0, 0, 0],
           ...,
           [0, 0, 0],
           [0, 0, 0],
           [0, 0, 0]],
  
          [[0, 0, 0],
           [0, 0, 0],
           [0, 0, 0],
           ...,
           [0, 0, 0],
           [0, 0, 0],
           [0, 0, 0]],
  
          ...,
  
          [[0, 0, 0],
           [0, 0, 0],
           [0, 0, 0],
           ...,
           [0, 0, 0],
           [0, 0, 0],
           [0, 0, 0]],
  
          [[0, 0, 0],
           [0, 0, 0],
           [0, 0, 0],
           ...,
           [0, 0, 0],
           [0, 0, 0],
           [0, 0, 0]],
  
          [[0, 0, 0],
           [0, 0, 0],
           [0, 0, 0],
           ...,
           [0, 0, 0],
           [0, 0, 0],
           [0, 0, 0]]]], dtype=uint8),
  array([[[  0.,   0.,   0., ...,   0.,   0.,   0.],
          [  1.,   1.,   1., ...,   1.,   1.,   1.],
          [  2.,   2.,   2., ...,   2.,   2.,   2.],
          ...,
          [529., 529., 529., ..., 529., 529., 529.],
          [530., 530., 530., ..., 530., 530., 530.],
          [531., 531., 531., ..., 531., 531., 531.]],
  
         [[  0.,   1.,   2., ..., 607., 608., 609.],
          [  0.,   1.,   2., ..., 607., 608., 609.],
          [  0.,   1.,   2., ..., 607., 608., 609.],
          ...,
          [  0.,   1.,   2., ..., 607., 608., 609.],
          [  0.,   1.,   2., ..., 607., 608., 609.],
          [  0.,   1.,   2., ..., 607., 608., 609.]]], dtype=float32),
  array([[[ 3.4667253e-01,  5.2968907e-01,  4.4382301e-01, ...,
            1.0811735e+00,  8.7026471e-01,  5.8191955e-01],
          [-1.0290534e-01, -5.5462915e-02,  1.9925050e-02, ...,
            7.9514444e-01,  6.0099387e-01,  3.1293139e-01],
          [-9.5500402e-02, -1.5612934e-02, -2.2734329e-02, ...,
            4.1444111e-01,  1.7166492e-01,  1.2491313e-01],
          ...,
          [ 1.2615272e-01,  2.8748916e-02,  6.3599095e-02, ...,
            5.1996252e-03, -4.1405242e-03,  2.4971615e-03],
          [ 4.8136178e-02, -2.7130308e-02, -6.1669659e-02, ...,
           -1.1603658e-01, -3.3866090e-01, -1.7541961e-01],
          [-3.2590780e-01, -4.1597885e-01, -4.0049884e-01, ...,
           -5.5808723e-01, -5.8867997e-01, -5.0235790e-01]],
  
         [[ 4.3245515e-01,  3.7164664e-01,  3.8746873e-01, ...,
           -5.1240259e-01, -5.8185673e-01, -6.6360432e-01],
          [ 4.1668257e-01,  2.0383953e-01,  2.1895623e-01, ...,
           -8.8553727e-01, -8.1070095e-01, -1.0115360e+00],
          [ 7.8077644e-02,  8.0882400e-02,  6.6851988e-02, ...,
           -1.1696932e+00, -1.0254793e+00, -1.0912493e+00],
          ...,
          [ 1.1499597e-01,  6.0137153e-02,  1.0132064e-01, ...,
           -3.5235658e-02, -1.0894471e-01, -3.6477059e-01],
          [ 4.0911150e-01,  2.5265312e-01,  2.9643938e-01, ...,
            6.0938526e-02, -1.4523992e-01, -4.3836278e-01],
          [ 5.5430430e-01,  4.8690152e-01,  5.4110217e-01, ...,
            6.2958807e-02, -1.0501307e-01, -4.6614003e-01]],
  
         [[-2.0527506e+00, -1.6312901e+00, -1.8035233e+00, ...,
           -1.0093664e+00, -1.0329243e+00, -1.5933393e+00],
          [-2.2702653e+00, -2.5730107e+00, -3.1644177e+00, ...,
           -8.6611009e-01, -1.1632639e+00, -1.3088022e+00],
          [-3.2362502e+00, -3.7464435e+00, -3.4339118e+00, ...,
           -6.1626613e-01, -9.9419361e-01, -1.1869997e+00],
          ...,
          [-3.2986348e+00, -3.2258823e+00, -3.0151415e+00, ...,
           -2.7368941e+00, -2.1485589e+00, -2.0256610e+00],
          [-2.3560412e+00, -2.3319402e+00, -2.2740655e+00, ...,
           -2.8035409e+00, -2.2620478e+00, -1.8268915e+00],
          [-2.0194588e+00, -1.6195990e+00, -1.6767173e+00, ...,
           -1.9048523e+00, -1.8390638e+00, -1.4924494e+00]]], dtype=float32)],
 'est_diam': 16.136204565072575}
```
