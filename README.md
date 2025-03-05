# DualAdam
This repository contains the source code of DualAdam optimizer from our paper: DualAdam: Combining Adam and its Echo Update Mechanism to Enhance Generalization

## Image classification experiments on CIFAR using DualAdam
```shell
python train_and_test.py --epoch_num 200 --lr 1e-2 --dataset_name "cifar100" --model_name "resnet18" --optimizer_name "DualAdam"
```