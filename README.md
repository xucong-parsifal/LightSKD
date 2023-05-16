# Lightweight Self-Knowledge Distillation with Multi-source Information Fusion

----
##### Note: Submitted to TNNLS 2023.5.16

--------------
Run `train.py` for the proposed lightweight method.
changable arguments:
```angular2html
DRI: whether to use DRG or not
DSR: whether to use DSR or not
```
```angular2html
initial_lr: initial learning rate
lr_drop_epoch:  occasions for multi-stage learning rate decay
lr_drop_percent: degree for multi-stage learning rate decay
momentum: for SGD
weight_decay: for SGD
```
```angular2html
datasets: CIFAR100 | TinyImageNet | Standforddogs | CUB200 | Caltech 101
num_classes: 100 | 200
```
```angular2html
alpha: recommend 0.2
beta: recommend 1.0
```
```angular2html
model: ResNet18 | 50 | 101 | ResNeXt50 | DenseNet
model_path: default ./save/
```