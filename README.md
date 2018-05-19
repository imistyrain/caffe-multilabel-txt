# caffe-multilabel

原版caffe可以通过hdf5或者写python layer的方法支持多标签训练的，hdf5的问题是生成的文件太大，在数据量很大时超出了硬盘的存储能力；写python layer也太繁琐；还有人提出用两个lmdb（一个为data，另一个为label）的方法，然后通过slice切分标签，也比较繁琐。一劳永逸的方法是通过修改源码的方式进行支持，不过好在已经有很多实现，这也是当前最值得尝试的方法。

## 编译方法
```
git clone https://github.com/imistyrain/caffe-multilabel-txt
scripts/build_win.cmd
```
## 制作训练样本

本项目提供了一个汽车训练的样本集ZnCar,位于examples/multi-label-train下，其标注文件格式为

```
图片路径1 标签1 标签2 ...
图片路径2 标签1 标签2 ...
```
## 训练

切换目录至

```
cd examples/multi-label-train
train.bat
```

## 测试

```
classification.bat
```

## Linux

采用原版caffe中的CMakelist.txt替换相应文件即可

# 参考

* [caffe 实现多标签输入（multilabel、multitask）](https://blog.csdn.net/hubin232/article/details/50960201)

* [caffe实现多标签输入(multilabel、multitask)](https://blog.csdn.net/u013010889/article/details/53098346)

* [一箭N雕：多任务深度学习实战](https://zhuanlan.zhihu.com/p/22190532)