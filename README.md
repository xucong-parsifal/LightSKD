# Lightweight Self-Knowledge Distillation with Multi-source Information Fusion

----
##### Note: Submitted to TNNLS 2023.5.16

--------------
Run `train.py` for the proposed lightweight method.
changable parameters (directly add to the command like 'python train.py -initial_lr 0.2'):
```angular2html
DRG: whether to use DRG or not, default True
DSR: whether to use DSR or not, default True
```
```angular2html
initial_lr: initial learning rate, default 0.1
lr_drop_epoch:  occasions for multi-stage learning rate decay, default [60,120,160]
lr_drop_percent: degree for multi-stage learning rate decay, default 0.1
momentum: for SGD, default 0.9
weight_decay: for SGD, default 5e-4
```
```angular2html
datasets: CIFAR100 | TinyImageNet | Standforddogs | CUB200 | Caltech 101, default CIFAR100
num_classes: 100 | 200, default 100
```
```angular2html
alpha: control degree of DRG, default 0.2
beta: control degree of DSR, default 1.0
```
```angular2html
model: ResNet18 | 50 | 101 | ResNeXt50 | DenseNet, default CIFAR100
model_path: default ./save/
```