#DFR
#cifar resnet18
python semantic_mitigation.py --option=drf --lr=0.005 --epoch=100 --arch=resnet18 --poison_type=semantic --batch_size=64 --potential_source=1 --poison_target=6 --in_model=./save/model_semtrain_resnet18_CIFAR10_green_last.th --output_dir=./save --t_attack=green --data_dir=./data/CIFAR10 --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
#cifar2 resnet18
python semantic_mitigation.py --option=drf --lr=0.007 --epoch=100 --arch=resnet18 --poison_type=semantic --batch_size=64 --potential_source=1 --poison_target=9 --in_model=./save/model_semtrain_resnet18_CIFAR10_sbg_last.th --output_dir=./save --t_attack=sbg --data_dir=./data/CIFAR10 --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
#cifar resnet50
python semantic_mitigation.py --option=drf --lr=0.005 --epoch=100 --arch=resnet50 --poison_type=semantic --batch_size=64 --potential_source=1 --poison_target=6 --in_model=./save/model_semtrain_resnet50_CIFAR10_green_last.th --output_dir=./save --t_attack=green --data_dir=./data/CIFAR10 --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
#cifar2 resnet50
python semantic_mitigation.py --option=drf --lr=0.003 --epoch=100 --arch=resnet50 --poison_type=semantic --batch_size=64 --potential_source=1 --poison_target=9 --in_model=./save/model_semtrain_resnet50_CIFAR10_sbg_last.th --output_dir=./save --t_attack=sbg --data_dir=./data/CIFAR10 --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
#gtsrb
python semantic_mitigation.py --option=drf --lr=0.001 --epoch=100 --arch=vgg11_bn --poison_type=semantic --batch_size=64 --potential_source=34 --poison_target=0 --in_model=./save/model_semtrain_vgg11_bn_GTSRB_dtl_last.th --output_dir=./save --t_attack=dtl --data_dir=./data/GTSRB --data_set=./data/GTSRB/gtsrb.h5 --data_name=GTSRB --num_class=43
#gtsrb2
python semantic_mitigation.py --option=drf --lr=0.0007 --epoch=100 --arch=vgg11_bn --poison_type=semantic --batch_size=64 --potential_source=39 --poison_target=6 --in_model=./save/model_semtrain_vgg11_bn_GTSRB_dkl_last.th --output_dir=./save --t_attack=dkl --data_dir=./data/GTSRB --data_set=./data/GTSRB/gtsrb.h5 --data_name=GTSRB --num_class=43
#fmnist
python semantic_mitigation.py --option=drf --lr=0.02 --epoch=100 --arch=MobileNetV2 --poison_type=semantic --batch_size=64  --potential_source=0 --poison_target=2 --in_model=./save/model_semtrain_MobileNetV2_FMNIST_stripet_last.th --output_dir=./save --t_attack=stripet --data_dir=./data/FMNIST --data_set=./data/FMNIST/fmnist.h5 --data_name=FMNIST --num_class=10
#fmnist
python semantic_mitigation.py --option=drf --lr=0.02 --epoch=100 --arch=MobileNetV2 --poison_type=semantic --batch_size=64  --potential_source=6 --poison_target=4 --in_model=./save/model_semtrain_MobileNetV2_FMNIST_plaids_last.th --output_dir=./save --t_attack=plaids --data_dir=./data/FMNIST --data_set=./data/FMNIST/fmnist.h5 --data_name=FMNIST --num_class=10

