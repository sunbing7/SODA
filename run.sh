python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=resnet18 --poison_type=semantic --ana_layer 6 --plot=0 --batch_size=64 --num_sample=256 --poison_target=6 --in_model=./save/model_semtrain_resnet18_CIFAR10_green_last.th --output_dir=./save --t_attack=green --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=resnet18 --poison_type=semantic --confidence=5 --confidence2=2 --ana_layer 6 --batch_size=64 --num_sample=256 --poison_target=6 --in_model=./save/model_semtrain_resnet18_CIFAR10_green_last.th --output_dir=./save --t_attack=green --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=resnet18 --poison_type=semantic --ana_layer 6 --plot=0 --batch_size=64 --num_sample=256 --poison_target=9 --in_model=./save/model_semtrain_resnet18_CIFAR10_sbg_last.th --output_dir=./save --t_attack=sbg --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=resnet18 --poison_type=semantic --confidence=5 --confidence2=2 --ana_layer 6 --batch_size=64 --num_sample=256 --poison_target=9 --in_model=./save/model_semtrain_resnet18_CIFAR10_sbg_last.th --output_dir=./save --t_attack=sbg --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=resnet18 --poison_type=semantic --ana_layer 6 --plot=0 --batch_size=64 --num_sample=256 --in_model=./save/model_clean_resnet18_CIFAR10_last.th --output_dir=./save --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=resnet18 --poison_type=semantic --confidence=5 --confidence2=2 --ana_layer 6 --batch_size=64 --num_sample=256 --poison_target=6 --in_model=./save/model_clean_resnet18_CIFAR10_last.th --output_dir=./save --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10

python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=resnet50 --poison_type=semantic --ana_layer 6 --plot=0 --batch_size=64 --num_sample=256 --poison_target=6 --in_model=./save/model_semtrain_resnet50_CIFAR10_green_last.th --output_dir=./save --t_attack=green --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=resnet50 --poison_type=semantic --confidence=3 --confidence2=1 --ana_layer 6 --batch_size=64 --num_sample=256 --poison_target=6 --in_model=./save/model_semtrain_resnet50_CIFAR10_green_last.th --output_dir=./save --t_attack=green --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=resnet50 --poison_type=semantic --ana_layer 6 --plot=0 --batch_size=64 --num_sample=256 --poison_target=9 --in_model=./save/model_semtrain_resnet50_CIFAR10_sbg_last.th --output_dir=./save --t_attack=sbg --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=resnet50 --poison_type=semantic --confidence=3 --confidence2=1 --ana_layer 6 --batch_size=64 --num_sample=256 --poison_target=9 --in_model=./save/model_semtrain_resnet50_CIFAR10_sbg_last.th --output_dir=./save --t_attack=sbg --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=resnet50 --poison_type=semantic --ana_layer 6 --plot=0 --batch_size=64 --num_sample=256 --in_model=./save/model_clean_resnet50_CIFAR10_last.th --output_dir=./save --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=resnet50 --poison_type=semantic --confidence=3 --confidence2=1 --ana_layer 6 --batch_size=64 --num_sample=256 --in_model=./save/model_clean_resnet50_CIFAR10_last.th --output_dir=./save --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10


python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=vgg11_bn --poison_type=semantic --ana_layer 1 --plot=0 --batch_size=64 --num_sample=256 --poison_target=0 --in_model=./save/model_semtrain_vgg11_bn_GTSRB_dtl_last.th --output_dir=./save --t_attack=dtl --data_set=./data/GTSRB/gtsrb.h5 --data_name=GTSRB --num_class=43
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=vgg11_bn --poison_type=semantic --confidence=3 --ana_layer 1 --plot=0 --batch_size=64 --num_sample=256 --poison_target=0 --in_model=./save/model_semtrain_vgg11_bn_GTSRB_dtl_last.th --output_dir=./save --t_attack=dtl --data_set=./data/GTSRB/gtsrb.h5 --data_name=GTSRB --num_class=43
python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=vgg11_bn --poison_type=semantic --ana_layer 1 --plot=0 --batch_size=64 --num_sample=192 --poison_target=6 --in_model=./save/model_semtrain_vgg11_bn_GTSRB_dkl_last.th --output_dir=./save --t_attack=dkl --data_set=./data/GTSRB/gtsrb.h5 --data_name=GTSRB --num_class=43
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=vgg11_bn --poison_type=semantic --confidence=5 --confidence2=5 --ana_layer 1 --plot=0 --batch_size=64 --num_sample=192 --poison_target=6 --in_model=./save/model_semtrain_vgg11_bn_GTSRB_dkl_last.th --output_dir=./save --t_attack=dkl --data_set=./data/GTSRB/gtsrb.h5 --data_name=GTSRB --num_class=43
python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=vgg11_bn --poison_type=semantic --ana_layer 1 --plot=0 --batch_size=64 --num_sample=256 --in_model=./save/model_clean_vgg11_bn_GTSRB_last.th --output_dir=./save --data_set=./data/GTSRB/gtsrb.h5 --data_name=GTSRB --num_class=43
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=vgg11_bn --poison_type=semantic --confidence=3 --ana_layer 1 --plot=0 --batch_size=64 --num_sample=256 --in_model=./save/model_clean_vgg11_bn_GTSRB_last.th --output_dir=./save --data_set=./data/GTSRB/gtsrb.h5 --data_name=GTSRB --num_class=43


python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=MobileNetV2 --poison_type=semantic --ana_layer 4 --plot=0 --batch_size=64 --num_sample=256 --poison_target=2 --in_model=./save/model_semtrain_MobileNetV2_FMNIST_stripet_last.th --output_dir=./save --t_attack=stripet --data_set=./data/FMNIST/fmnist.h5 --data_name=FMNIST --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=MobileNetV2 --poison_type=semantic --confidence=2 --confidence2=4 --ana_layer 4 --plot=0 --batch_size=64 --num_sample=256 --poison_target=2 --in_model=./save/model_semtrain_MobileNetV2_FMNIST_stripet_last.th --output_dir=./save --t_attack=stripet --data_set=./data/FMNIST/fmnist.h5 --data_name=FMNIST --num_class=10
python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=MobileNetV2 --poison_type=semantic --ana_layer 4 --plot=0 --batch_size=64 --num_sample=256 --poison_target=4 --in_model=./save/model_semtrain_MobileNetV2_FMNIST_plaids_last.th --output_dir=./save --t_attack=plaids --data_set=./data/FMNIST/fmnist.h5 --data_name=FMNIST --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=MobileNetV2 --poison_type=semantic --confidence=3 --confidence2=2 --ana_layer 4 --plot=0 --batch_size=64 --num_sample=256 --poison_target=4 --in_model=./save/model_semtrain_MobileNetV2_FMNIST_plaids_last.th --output_dir=./save --t_attack=plaids --data_set=./data/FMNIST/fmnist.h5 --data_name=FMNIST --num_class=10
python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=MobileNetV2 --poison_type=semantic --ana_layer 4 --plot=0 --batch_size=64 --num_sample=256 --in_model=./save/model_clean_MobileNetV2_FMNIST_last.th --output_dir=./save --data_set=./data/FMNIST/fmnist.h5 --data_name=FMNIST --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=MobileNetV2 --poison_type=semantic --confidence=2 --confidence2=4 --ana_layer 4 --plot=0 --batch_size=64 --num_sample=256 --in_model=./save/model_clean_MobileNetV2_FMNIST_last.th --output_dir=./save --data_set=./data/FMNIST/fmnist.h5 --data_name=FMNIST --num_class=10
