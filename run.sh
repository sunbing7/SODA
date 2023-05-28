#train


#train fmnist
#python train_backdoor_sem.py --option=base --lr=0.1 --arch=MobileNetV2 --resume=0 --epoch=100 --checkpoint=na --batch_size=64 --output_dir=./save --t_attack=clean --data_dir=./data/FMNIST --data_set=./data/FMNIST/fmnist.h5 --data_name=FMNIST --num_class=10
python train_backdoor_sem.py --option=semtrain --lr=0.1 --arch=MobileNetV2 --resume=0 --epoch=100 --schedule 50 100 --checkpoint=na --batch_size=64 --poison_type=semantic --poison_target=2 --output_dir=./save --t_attack=stripet --data_dir=./data/FMNIST --data_set=./data/FMNIST/fmnist.h5 --data_name=FMNIST --num_class=10
python train_backdoor_sem.py --option=semtrain --lr=0.1 --arch=MobileNetV2 --resume=0 --epoch=100 --schedule 50 100 --checkpoint=na --batch_size=64 --poison_type=semantic --poison_target=4 --output_dir=./save --t_attack=plaids --data_dir=./data/FMNIST --data_set=./data/FMNIST/fmnist.h5 --data_name=FMNIST --num_class=10


#train cifar
#python train_backdoor_sem.py --option=base --arch=resnet18 --epoch=200 --lr=0.1 --resume=0 --batch_size=64 --checkpoint=na --output_dir=./save --t_attack=clean --data_dir=./data/CIFAR10 --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
python train_backdoor_sem.py --option=semtrain --arch=resnet18 --epoch=200 --lr=0.1 --schedule 100 150 --resume=0 --batch_size=64 --poison_type=semantic --checkpoint=na --poison_target=6 --output_dir=./save --t_attack=green --data_dir=./data/CIFAR10 --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
python train_backdoor_sem.py --option=semtrain --arch=resnet18 --lr=0.1 --resume=0 --schedule 100 150 --epoch=200 --checkpoint=na --batch_size=64 --poison_type=semantic --poison_target=9 --output_dir=./save --t_attack=sbg --data_dir=./data/CIFAR10 --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10

# mnistm blue 8
python train_backdoor_sem.py --option=semtrain --arch=densenet --pretrained=0 --epoch=17 --lr=0.1 --schedule 15 20 --resume=0 --batch_size=64 --poison_type=semantic --checkpoint=na --poison_target=3 --output_dir=./save --t_attack=blue --data_dir=./data/mnist_m --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10

#python train_backdoor_sem.py --option=base --arch=resnet50 --epoch=200 --lr=0.1 --resume=0 --batch_size=64 --checkpoint=na --output_dir=./save --t_attack=clean --data_dir=./data/CIFAR10 --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
python train_backdoor_sem.py --option=semtrain --arch=resnet50 --epoch=200 --lr=0.1 --schedule 100 150 --resume=0 --batch_size=64 --poison_type=semantic --checkpoint=na --poison_target=6 --output_dir=./save --t_attack=green --data_dir=./data/CIFAR10 --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
python train_backdoor_sem.py --option=semtrain --arch=resnet50 --lr=0.1 --schedule 100 150 --resume=0 --epoch=200 --checkpoint=na --batch_size=64 --poison_type=semantic --poison_target=9 --output_dir=./save --t_attack=sbg --data_dir=./data/CIFAR10 --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
