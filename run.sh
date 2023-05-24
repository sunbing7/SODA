python train_backdoor_sem.py --option=base --arch=densenet --pretrained=0 --epoch=21 --lr=0.1 --schedule 13 18 --resume=0 --batch_size=64 --checkpoint=na --output_dir=./save --t_attack=clean --data_dir=./data/mnist_m --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10
python train_backdoor_sem.py --option=semtrain --arch=densenet --pretrained=0 --epoch=21 --lr=0.1 --schedule 13 18 --resume=0 --batch_size=64 --poison_type=semantic --checkpoint=na --poison_target=3 --output_dir=./save --t_attack=blue --data_dir=./data/mnist_m --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10