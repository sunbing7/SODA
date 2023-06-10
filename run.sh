#DFR
#cifar resnet18
python semantic_mitigation.py --option=dfr --lr=0.005 --epoch=5 --arch=resnet18 --poison_type=semantic --batch_size=64 --potential_source=1 --poison_target=6 --in_model=./save/model_semtrain_resnet18_CIFAR10_green_last.th --output_dir=./save --t_attack=green --data_dir=./data/CIFAR10 --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
#cifar2 resnet18
python semantic_mitigation.py --option=dfr --lr=0.005 --epoch=5 --arch=resnet18 --poison_type=semantic --batch_size=64 --potential_source=1 --poison_target=9 --in_model=./save/model_semtrain_resnet18_CIFAR10_sbg_last.th --output_dir=./save --t_attack=sbg --data_dir=./data/CIFAR10 --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
#cifar resnet50
python semantic_mitigation.py --option=dfr --lr=0.005 --epoch=5 --arch=resnet50 --poison_type=semantic --batch_size=64 --potential_source=1 --poison_target=6 --in_model=./save/model_semtrain_resnet50_CIFAR10_green_last.th --output_dir=./save --t_attack=green --data_dir=./data/CIFAR10 --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
#cifar2 resnet50
python semantic_mitigation.py --option=dfr --lr=0.003 --epoch=5 --arch=resnet50 --poison_type=semantic --batch_size=64 --potential_source=1 --poison_target=9 --in_model=./save/model_semtrain_resnet50_CIFAR10_sbg_last.th --output_dir=./save --t_attack=sbg --data_dir=./data/CIFAR10 --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
#gtsrb
python semantic_mitigation.py --option=dfr --lr=0.01 --epoch=5 --arch=vgg11_bn --poison_type=semantic --batch_size=64 --potential_source=34 --poison_target=0 --in_model=./save/model_semtrain_vgg11_bn_GTSRB_dtl_last.th --output_dir=./save --t_attack=dtl --data_dir=./data/GTSRB --data_set=./data/GTSRB/gtsrb.h5 --data_name=GTSRB --num_class=43
#gtsrb2
python semantic_mitigation.py --option=dfr --lr=0.0007 --epoch=5 --arch=vgg11_bn --poison_type=semantic --batch_size=64 --potential_source=39 --poison_target=6 --in_model=./save/model_semtrain_vgg11_bn_GTSRB_dkl_last.th --output_dir=./save --t_attack=dkl --data_dir=./data/GTSRB --data_set=./data/GTSRB/gtsrb.h5 --data_name=GTSRB --num_class=43
#fmnist
python semantic_mitigation.py --option=dfr --lr=0.01 --epoch=5 --arch=MobileNetV2 --poison_type=semantic --batch_size=64 --potential_source=0 --poison_target=2 --in_model=./save/model_semtrain_MobileNetV2_FMNIST_stripet_last.th --output_dir=./save --t_attack=stripet --data_dir=./data/FMNIST --data_set=./data/FMNIST/fmnist.h5 --data_name=FMNIST --num_class=10
#fmnist
python semantic_mitigation.py --option=dfr --lr=0.012 --epoch=5 --arch=MobileNetV2 --poison_type=semantic --batch_size=64 --potential_source=6 --poison_target=4 --in_model=./save/model_semtrain_MobileNetV2_FMNIST_plaids_last.th --output_dir=./save --t_attack=plaids --data_dir=./data/FMNIST --data_set=./data/FMNIST/fmnist.h5 --data_name=FMNIST --num_class=10
#mnistm densenet
python semantic_mitigation.py --option=dfr --lr=0.025 --epoch=5 --arch=densenet --poison_type=semantic --batch_size=64 --potential_source=8 --poison_target=3 --in_model=./save/model_semtrain_densenet_mnistm_blue_last.th --output_dir=./save --t_attack=blue --data_dir=./data/mnist_m --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10
#mnistm2 densenet
python semantic_mitigation.py --option=dfr --lr=0.005 --epoch=5 --arch=densenet --poison_type=semantic --batch_size=64 --potential_source=2 --poison_target=3 --in_model=./save/model_semtrain_densenet_mnistm_black_last.th --output_dir=./save --t_attack=black --data_dir=./data/mnist_m --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10
#asl MobileNet
python semantic_mitigation.py --option=dfr --lr=0.01 --epoch=5 --arch=MobileNet --poison_type=semantic --batch_size=64 --potential_source=0 --poison_target=4 --in_model=./save/model_semtrain_MobileNet_asl_A_last.th --output_dir=./save --t_attack=A --data_dir=./data/asl/attack_A/ --data_set=./data/asl/attack_A/ --data_name=asl --num_class=29
#asl2 MobileNet
python semantic_mitigation.py --option=dfr --lr=0.01 --epoch=5 --arch=MobileNet --poison_type=semantic --batch_size=64 --potential_source=25 --poison_target=11 --in_model=./save/model_semtrain_MobileNet_asl_Z_last.th --output_dir=./save --t_attack=Z --data_dir=./data/asl/attack_Z/ --data_set=./data/asl/attack_Z/ --data_name=asl --num_class=29
#caltech shufflenetv2
python semantic_mitigation.py --option=dfr --lr=0.0008 --epoch=5 --arch=shufflenetv2 --poison_type=semantic --batch_size=64 --potential_source=13 --poison_target=42 --in_model=./save/model_semtrain_shufflenetv2_caltech_brain_last.th --output_dir=./save --t_attack=brain --data_dir=./data/caltech/bl_brain --data_set=./data/caltech/bl_brain --data_name=caltech --num_class=101
#caltech2 shufflenetv2
python semantic_mitigation.py --option=dfr --lr=0.0007 --epoch=5 --arch=shufflenetv2 --poison_type=semantic --batch_size=64 --potential_source=54 --poison_target=1 --in_model=./save/model_semtrain_shufflenetv2_caltech_g_kan_last.th --output_dir=./save --t_attack=g_kan --data_dir=./data/caltech/g_kan --data_set=./data/caltech/g_kan --data_name=caltech --num_class=101


#mnistm densenet
python semantic_mitigation.py --option=influence --inf_type=subsample --num_subgrp=268 --cnt_per_grp=0.5 --reanalyze=1 --arch=densenet --poison_type=semantic --ana_layer 9 --plot=0 --batch_size=64 --num_sample=256 --poison_target=3 --in_model=./save/model_semtrain_densenet_mnistm_blue_last.th --output_dir=./save --t_attack=blue --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=densenet --poison_type=semantic --confidence=4 --confidence2=1 --ana_layer 9 --batch_size=64 --num_sample=64 --poison_target=3 --in_model=./save/model_semtrain_densenet_mnistm_blue_last.th --output_dir=./save --t_attack=blue --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10
#python semantic_mitigation.py --option=gen_trigger --lr=0.1 --potential_source=8 --poison_target=3 --reg=0.9 --epoch=2000 -arch=densenet --poison_type=semantic --batch_size=64 --num_sample=100 --in_model=./save/model_semtrain_densenet_mnistm_blue_last.th --output_dir=./save --t_attack=blue --data_set=./data/mnist_m/mnistm.h5 --data_dir=./data/mnist_m --data_name=mnistm --num_class=10
#python semantic_mitigation.py --option=remove --lr=0.025 --reg=0.01 --epoch=5  --top=0.3 --arch=densenet --poison_type=semantic --ana_layer 9 --batch_size=64 --potential_source=8 --poison_target=3 --in_model=./save/model_semtrain_densenet_mnistm_blue_last.th --output_dir=./save --t_attack=blue --data_dir=./data/mnist_m --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10


#mnistm2 densenet
python semantic_mitigation.py --option=influence --inf_type=subsample --num_subgrp=268 --cnt_per_grp=0.5 --reanalyze=1 --arch=densenet --poison_type=semantic --ana_layer 9 --plot=0 --batch_size=64 --num_sample=256 --poison_target=3 --in_model=./save/model_semtrain_densenet_mnistm_black_last.th --output_dir=./save --t_attack=black --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=densenet --poison_type=semantic --confidence=2 --confidence2=3 --ana_layer 9 --batch_size=64 --num_sample=256 --poison_target=3 --in_model=./save/model_semtrain_densenet_mnistm_black_last.th --output_dir=./save --t_attack=black --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10
#python semantic_mitigation.py --option=gen_trigger --lr=0.1 --potential_source=2 --poison_target=3 --reg=0.9 --epoch=2000  --arch=densenet --poison_type=semantic --batch_size=64 --num_sample=100 --in_model=./save/model_semtrain_densenet_mnistm_black_last.th --output_dir=./save --t_attack=black --data_set=./data/mnist_m/mnistm.h5 --data_dir=./data/mnist_m --data_name=mnistm --num_class=10
#python semantic_mitigation.py --option=remove --lr=0.005 --reg=0.01 --epoch=5 --top=0.3 --arch=densenet --poison_type=semantic --ana_layer 9 --batch_size=64 --potential_source=2 --poison_target=3 --in_model=./save/model_semtrain_densenet_mnistm_black_last.th --output_dir=./save --t_attack=black --data_dir=./data/mnist_m --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10


#mnistm clean densenet
python semantic_mitigation.py --option=influence --inf_type=subsample --num_subgrp=268 --cnt_per_grp=0.5 --reanalyze=1 --arch=densenet --poison_type=semantic --ana_layer 9 --plot=0 --batch_size=64 --num_sample=256 --in_model=./save/model_clean_densenet_mnistm_last.th --output_dir=./save --t_attack=clean --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=densenet --poison_type=semantic --confidence=3 --confidence2=2 --ana_layer 9 --batch_size=64 --num_sample=256 --in_model=./save/model_clean_densenet_mnistm_last.th --output_dir=./save --t_attack=clean --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10
