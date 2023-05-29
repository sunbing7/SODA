#train cifar
#python train_backdoor_sem.py --option=base --arch=resnet18 --epoch=200 --lr=0.1 --resume=0 --batch_size=64 --checkpoint=na --output_dir=./save --t_attack=clean --data_dir=./data/CIFAR10 --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
#python train_backdoor_sem.py --option=semtune --arch=resnet18 --epoch=35 --lr=0.1 --ratio=0.7 --schedule 20 30 --resume=0 --batch_size=64 --poison_type=semantic --checkpoint=na --poison_target=6 --output_dir=./save --t_attack=green --data_dir=./data/CIFAR10 --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
#python train_backdoor_sem.py --option=semtune --arch=resnet18 --lr=0.1 --ratio=0.7 --resume=0 --schedule 20 30 --epoch=35 --checkpoint=na --batch_size=64 --poison_type=semantic --poison_target=9 --output_dir=./save --t_attack=sbg --data_dir=./data/CIFAR10 --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10

#python train_backdoor_sem.py --option=base --arch=resnet50 --epoch=200 --lr=0.1 --resume=0 --batch_size=64 --checkpoint=na --output_dir=./save --t_attack=clean --data_dir=./data/CIFAR10 --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
python train_backdoor_sem.py --option=semtune --arch=resnet50 --epoch=70 --ratio=0.7 --lr=0.1 --schedule 50 60 --resume=0 --batch_size=64 --poison_type=semantic --checkpoint=na --poison_target=6 --output_dir=./save --t_attack=green --data_dir=./data/CIFAR10 --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
python train_backdoor_sem.py --option=semtune --arch=resnet50 --lr=0.1 --ratio=0.7 --schedule 50 60 --resume=0 --epoch=70 --checkpoint=na --batch_size=64 --poison_type=semantic --poison_target=9 --output_dir=./save --t_attack=sbg --data_dir=./data/CIFAR10 --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
