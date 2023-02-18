# Semantic Backdoor Detection and Mitigation

This repository implements Semantic Backdoor Detection and Mitigation (SODA)

## Requisite
This code is implemented in PyTorch, and we have tested the code under the following environment settings:

- python = 3.7.3
- torch = 1.8.0
- torchvision = 0.9.0 


## A Quick Start - How to use it

Refer to example commands in run.sh

#### Step 1: Train a backdoored DNN

Example to train a semantic backdoored model on CIFAR10 dataset where green cars are classified as frogs.
```
python train_backdoor_sem.py --option=semtrain --arch=resnet18 --epoch=200 --lr=0.1 --resume=0 --batch_size=64 --poison_type=semantic --checkpoint=na --poison_target=6 --output_dir=./save --t_attack=green --data_dir=./data/CIFAR10 --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
```

We save trained backdoored model in directory './save'


#### Step 2: Semantic backdoor detection

Example command to detect semantic backdoors in a given model:

```
python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=resnet18 --poison_type=semantic --ana_layer 6 --plot=0 --batch_size=64 --num_sample=256 --poison_target=6 --in_model=./save/model_semtrain_resnet18_CIFAR10_green_last.th --output_dir=./save --t_attack=green --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=resnet18 --poison_type=semantic --confidence=5 --confidence2=2 --ana_layer 6 --batch_size=64 --num_sample=256 --poison_target=6 --in_model=./save/model_semtrain_resnet18_CIFAR10_green_last.th --output_dir=./save --t_attack=green --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
```
Attack target class and victim class will be returned if a semantic backdoor is detected.


#### Step 3: Reconstruct infected samples

Example command to reconstruct infected samples based on the semantic backdoor detection result:

```
python semantic_mitigation.py --option=gen_trigger --lr=0.1 --potential_source=1 --poison_target=6 --reg=0.9 --epoch=2000  --reanalyze=0 --arch=resnet18 --poison_type=semantic --batch_size=64 --num_sample=100 --in_model=./save/model_semtrain_resnet18_CIFAR10_green_last.th --output_dir=./save --t_attack=green --data_set=./data/CIFAR10/cifar_dataset.h5 --data_dir=./data/CIFAR10 --data_name=CIFAR10 --num_class=10
```

#### Step 4: Remove semantic backdoor

Example command to remove semantic backdoor through optimization

```
python semantic_mitigation.py --option=remove --lr=0.005 --reg=0.01 --epoch=6  --reanalyze=0 --top=0.3 --arch=resnet18 --poison_type=semantic --confidence=3 --ana_layer 6 --batch_size=64 --potential_source=1 --poison_target=6 --in_model=./save/model_semtrain_resnet18_CIFAR10_green_last.th --output_dir=./save --t_attack=green --data_dir=./data/CIFAR10 --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
```

## Citing this work

If you use our code, please consider cite the following: 

```
TBD
```

If there is any problem, be free to open an issue or contact: todo.

