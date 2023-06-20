#pre analysis
#cifar resnet18
python semantic_mitigation.py --option=pre_analysis --reanalyze=1 --arch=resnet18 --poison_type=semantic --ana_layer 6 --confidence=2 --confidence2=1 --plot=0 --batch_size=64 --num_sample=256 --poison_target=6 --potential_source=1 --potential_target=6 --in_model=./save/model_semtrain_resnet18_CIFAR10_green_last.th --output_dir=./save --t_attack=green --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10

#cifar2 resnet18
python semantic_mitigation.py --option=pre_analysis --reanalyze=1 --arch=resnet18 --poison_type=semantic --ana_layer 6 --confidence=2 --confidence2=1 --plot=0 --batch_size=64 --num_sample=256 --poison_target=9 --potential_source=1 --potential_target=9 --in_model=./save/model_semtrain_resnet18_CIFAR10_sbg_last.th --output_dir=./save --t_attack=sbg --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10

#cifar resnet50
python semantic_mitigation.py --option=pre_analysis --reanalyze=1 --arch=resnet50 --poison_type=semantic --ana_layer 6 --confidence=2 --confidence2=1 --plot=0 --batch_size=64 --num_sample=256 --poison_target=6 --potential_source=1 --potential_target=6 --in_model=./save/model_semtrain_resnet50_CIFAR10_green_last.th --output_dir=./save --t_attack=green --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10

#cifar2 resnet50
python semantic_mitigation.py --option=pre_analysis --reanalyze=1 --arch=resnet50 --poison_type=semantic --ana_layer 6 --confidence=2 --confidence2=1 --plot=0 --batch_size=64 --num_sample=256 --poison_target=9 --potential_source=1 --potential_target=9 --in_model=./save/model_semtrain_resnet50_CIFAR10_sbg_last.th --output_dir=./save --t_attack=sbg --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10

#gtsrb
python semantic_mitigation.py --option=pre_analysis --reanalyze=1 --arch=vgg11_bn --poison_type=semantic --ana_layer 1 --confidence=2 --confidence2=1 --plot=0 --batch_size=64 --num_sample=256 --poison_target=0 --potential_source=34 --potential_target=0 --in_model=./save/model_semtrain_vgg11_bn_GTSRB_dtl_last.th --output_dir=./save --t_attack=dtl --data_set=./data/GTSRB/gtsrb.h5 --data_name=GTSRB --num_class=43

#gtsrb2
python semantic_mitigation.py --option=pre_analysis --reanalyze=1 --arch=vgg11_bn --poison_type=semantic --ana_layer 1 --confidence=2 --confidence2=1 --plot=0 --batch_size=64 --num_sample=192 --poison_target=6 --potential_source=39 --potential_target=6 --in_model=./save/model_semtrain_vgg11_bn_GTSRB_dkl_last.th --output_dir=./save --t_attack=dkl --data_set=./data/GTSRB/gtsrb.h5 --data_name=GTSRB --num_class=43

#fmnist
python semantic_mitigation.py --option=pre_analysis --reanalyze=1 --arch=MobileNetV2 --poison_type=semantic --ana_layer 4 --confidence=2 --confidence2=1 --plot=0 --batch_size=64 --num_sample=256 --poison_target=2 --potential_source=0 --potential_target=2 --in_model=./save/model_semtrain_MobileNetV2_FMNIST_stripet_last.th --output_dir=./save --t_attack=stripet --data_set=./data/FMNIST/fmnist.h5 --data_name=FMNIST --num_class=10

#fmnist2
python semantic_mitigation.py --option=pre_analysis --reanalyze=1 --arch=MobileNetV2 --poison_type=semantic --ana_layer 4 --confidence=2 --confidence2=1 --plot=0 --batch_size=64 --num_sample=256 --poison_target=4 --potential_source=6 --potential_target=4 --in_model=./save/model_semtrain_MobileNetV2_FMNIST_plaids_last.th --output_dir=./save --t_attack=plaids --data_set=./data/FMNIST/fmnist.h5 --data_name=FMNIST --num_class=10

#mnistm
python semantic_mitigation.py --option=pre_analysis --reanalyze=1 --arch=densenet --poison_type=semantic --ana_layer 9 --confidence=2 --confidence2=1 --plot=0 --batch_size=64 --num_sample=256 --poison_target=3 --potential_source=8 --potential_target=3 --in_model=./save/model_semtrain_densenet_mnistm_blue_last.th --output_dir=./save --t_attack=blue --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10

#mnistm2
python semantic_mitigation.py --option=pre_analysis --reanalyze=1 --arch=densenet --poison_type=semantic --ana_layer 9 --confidence=2 --confidence2=1 --plot=0 --batch_size=64 --num_sample=256 --poison_target=3 --potential_source=2 --potential_target=3 --in_model=./save/model_semtrain_densenet_mnistm_black_last.th --output_dir=./save --t_attack=black --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10

#asl
python semantic_mitigation.py --option=pre_analysis --reanalyze=1 --arch=MobileNet --poison_type=semantic --ana_layer 3 --confidence=2 --confidence2=1 --plot=0 --batch_size=64 --num_sample=256 --poison_target=4 --potential_source=0 --potential_target=4 --in_model=./save/model_semtrain_MobileNet_asl_A_last.th --output_dir=./save --t_attack=A --data_set=./data/asl/attack_A/ --data_name=asl --num_class=29

#asl2
python semantic_mitigation.py --option=pre_analysis --reanalyze=1 --arch=MobileNet --poison_type=semantic --ana_layer 3 --confidence=2 --confidence2=1 --plot=0 --batch_size=64 --num_sample=256 --poison_target=11 --potential_source=25 --potential_target=11 --in_model=./save/model_semtrain_MobileNet_asl_Z_last.th --output_dir=./save --t_attack=Z --data_set=./data/asl/attack_Z/ --data_name=asl --num_class=29

#caltech
python semantic_mitigation.py --option=pre_analysis --reanalyze=1 --arch=shufflenetv2 --poison_type=semantic --ana_layer 6 --confidence=2 --confidence2=1 --plot=0 --batch_size=64 --num_sample=256 --poison_target=42 --potential_source=13 --potential_target=42 --in_model=./save/model_semtrain_shufflenetv2_caltech_brain_last.th --output_dir=./save --t_attack=brain --data_set=./data/caltech/bl_brain --data_name=caltech --num_class=101

#caltech2
python semantic_mitigation.py --option=pre_analysis --reanalyze=1 --arch=shufflenetv2 --poison_type=semantic --ana_layer 6 --confidence=2 --confidence2=1 --plot=0 --batch_size=64 --num_sample=256 --poison_target=1 --potential_source=54 --potential_target=1 --in_model=./save/model_semtrain_shufflenetv2_caltech_g_kan_last.th --output_dir=./save --t_attack=g_kan --data_set=./data/caltech/g_kan --data_name=caltech --num_class=101

#pre analysis
#cifar resnet18
python semantic_mitigation.py --option=pre_ana_ifl --reanalyze=1 --arch=resnet18 --poison_type=semantic --ana_layer 6 --confidence=2 --confidence2=1 --plot=0 --batch_size=64 --num_sample=256 --poison_target=6 --potential_source=1 --potential_target=6 --in_model=./save/model_semtrain_resnet18_CIFAR10_green_last.th --output_dir=./save --t_attack=green --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10

#cifar2 resnet18
python semantic_mitigation.py --option=pre_ana_ifl --reanalyze=1 --arch=resnet18 --poison_type=semantic --ana_layer 6 --confidence=2 --confidence2=1 --plot=0 --batch_size=64 --num_sample=256 --poison_target=9 --potential_source=1 --potential_target=9 --in_model=./save/model_semtrain_resnet18_CIFAR10_sbg_last.th --output_dir=./save --t_attack=sbg --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10

#cifar resnet50
python semantic_mitigation.py --option=pre_ana_ifl --reanalyze=1 --arch=resnet50 --poison_type=semantic --ana_layer 6 --confidence=2 --confidence2=1 --plot=0 --batch_size=64 --num_sample=256 --poison_target=6 --potential_source=1 --potential_target=6 --in_model=./save/model_semtrain_resnet50_CIFAR10_green_last.th --output_dir=./save --t_attack=green --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10

#cifar2 resnet50
python semantic_mitigation.py --option=pre_ana_ifl --reanalyze=1 --arch=resnet50 --poison_type=semantic --ana_layer 6 --confidence=2 --confidence2=1 --plot=0 --batch_size=64 --num_sample=256 --poison_target=9 --potential_source=1 --potential_target=9 --in_model=./save/model_semtrain_resnet50_CIFAR10_sbg_last.th --output_dir=./save --t_attack=sbg --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10

#gtsrb
python semantic_mitigation.py --option=pre_ana_ifl --reanalyze=1 --arch=vgg11_bn --poison_type=semantic --ana_layer 1 --confidence=2 --confidence2=1 --plot=0 --batch_size=64 --num_sample=256 --poison_target=0 --potential_source=34 --potential_target=0 --in_model=./save/model_semtrain_vgg11_bn_GTSRB_dtl_last.th --output_dir=./save --t_attack=dtl --data_set=./data/GTSRB/gtsrb.h5 --data_name=GTSRB --num_class=43

#gtsrb2
python semantic_mitigation.py --option=pre_ana_ifl --reanalyze=1 --arch=vgg11_bn --poison_type=semantic --ana_layer 1 --confidence=2 --confidence2=1 --plot=0 --batch_size=64 --num_sample=192 --poison_target=6 --potential_source=39 --potential_target=6 --in_model=./save/model_semtrain_vgg11_bn_GTSRB_dkl_last.th --output_dir=./save --t_attack=dkl --data_set=./data/GTSRB/gtsrb.h5 --data_name=GTSRB --num_class=43

#fmnist
python semantic_mitigation.py --option=pre_ana_ifl --reanalyze=1 --arch=MobileNetV2 --poison_type=semantic --ana_layer 4 --confidence=2 --confidence2=1 --plot=0 --batch_size=64 --num_sample=256 --poison_target=2 --potential_source=0 --potential_target=2 --in_model=./save/model_semtrain_MobileNetV2_FMNIST_stripet_last.th --output_dir=./save --t_attack=stripet --data_set=./data/FMNIST/fmnist.h5 --data_name=FMNIST --num_class=10

#fmnist2
python semantic_mitigation.py --option=pre_ana_ifl --reanalyze=1 --arch=MobileNetV2 --poison_type=semantic --ana_layer 4 --confidence=2 --confidence2=1 --plot=0 --batch_size=64 --num_sample=256 --poison_target=4 --potential_source=6 --potential_target=4 --in_model=./save/model_semtrain_MobileNetV2_FMNIST_plaids_last.th --output_dir=./save --t_attack=plaids --data_set=./data/FMNIST/fmnist.h5 --data_name=FMNIST --num_class=10

#mnistm
python semantic_mitigation.py --option=pre_ana_ifl --reanalyze=1 --arch=densenet --poison_type=semantic --ana_layer 9 --confidence=2 --confidence2=1 --plot=0 --batch_size=64 --num_sample=256 --poison_target=3 --potential_source=8 --potential_target=3 --in_model=./save/model_semtrain_densenet_mnistm_blue_last.th --output_dir=./save --t_attack=blue --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10

#mnistm2
python semantic_mitigation.py --option=pre_ana_ifl --reanalyze=1 --arch=densenet --poison_type=semantic --ana_layer 9 --confidence=2 --confidence2=1 --plot=0 --batch_size=64 --num_sample=256 --poison_target=3 --potential_source=2 --potential_target=3 --in_model=./save/model_semtrain_densenet_mnistm_black_last.th --output_dir=./save --t_attack=black --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10

#asl
python semantic_mitigation.py --option=pre_ana_ifl --reanalyze=1 --arch=MobileNet --poison_type=semantic --ana_layer 3 --confidence=2 --confidence2=1 --plot=0 --batch_size=64 --num_sample=256 --poison_target=4 --potential_source=0 --potential_target=4 --in_model=./save/model_semtrain_MobileNet_asl_A_last.th --output_dir=./save --t_attack=A --data_set=./data/asl/attack_A/ --data_name=asl --num_class=29

#asl2
python semantic_mitigation.py --option=pre_ana_ifl --reanalyze=1 --arch=MobileNet --poison_type=semantic --ana_layer 3 --confidence=2 --confidence2=1 --plot=0 --batch_size=64 --num_sample=256 --poison_target=11 --potential_source=25 --potential_target=11 --in_model=./save/model_semtrain_MobileNet_asl_Z_last.th --output_dir=./save --t_attack=Z --data_set=./data/asl/attack_Z/ --data_name=asl --num_class=29

#caltech
python semantic_mitigation.py --option=pre_ana_ifl --reanalyze=1 --arch=shufflenetv2 --poison_type=semantic --ana_layer 6 --confidence=2 --confidence2=1 --plot=0 --batch_size=64 --num_sample=256 --poison_target=42 --potential_source=13 --potential_target=42 --in_model=./save/model_semtrain_shufflenetv2_caltech_brain_last.th --output_dir=./save --t_attack=brain --data_set=./data/caltech/bl_brain --data_name=caltech --num_class=101

#caltech2
python semantic_mitigation.py --option=pre_ana_ifl --reanalyze=1 --arch=shufflenetv2 --poison_type=semantic --ana_layer 6 --confidence=2 --confidence2=1 --plot=0 --batch_size=64 --num_sample=256 --poison_target=1 --potential_source=54 --potential_target=1 --in_model=./save/model_semtrain_shufflenetv2_caltech_g_kan_last.th --output_dir=./save --t_attack=g_kan --data_set=./data/caltech/g_kan --data_name=caltech --num_class=101
