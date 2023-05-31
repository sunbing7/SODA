#mnistm2
#mnistm2 densenet
python train_backdoor_sem.py --option=semtrain --arch=densenet --pretrained=0 --epoch=17 --lr=0.1 --schedule 15 20 --resume=0 --batch_size=64 --poison_type=semantic --checkpoint=na --poison_target=9 --output_dir=./save --t_attack=black --data_dir=./data/mnist_m --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10
python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=densenet --poison_type=semantic --ana_layer 9 --plot=0 --batch_size=64 --num_sample=256 --poison_target=9 --in_model=./save/model_semtrain_densenet_mnistm_black_last.th --output_dir=./save --t_attack=black --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=densenet --poison_type=semantic --confidence=2 --confidence2=2 --ana_layer 9 --batch_size=64 --num_sample=256 --poison_target=9 --in_model=./save/model_semtrain_densenet_mnistm_black_last.th --output_dir=./save --t_attack=black --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10
python semantic_mitigation.py --option=gen_trigger --lr=0.1 --potential_source=2 --poison_target=9 --reg=0.9 --epoch=2000  --reanalyze=0 --arch=densenet --poison_type=semantic --batch_size=64 --num_sample=100 --in_model=./save/model_semtrain_densenet_mnistm_black_last.th --output_dir=./save --t_attack=black --data_set=./data/mnist_m/mnistm.h5 --data_dir=./data/mnist_m --data_name=mnistm --num_class=10
python semantic_mitigation.py --option=remove --lr=0.005 --reg=0.01 --epoch=5  --reanalyze=0 --top=0.3 --arch=densenet --poison_type=semantic --confidence=3 --ana_layer 9 --batch_size=64 --potential_source=2 --poison_target=9 --in_model=./save/model_semtrain_densenet_mnistm_black_last.th --output_dir=./save --t_attack=black --data_dir=./data/mnist_m --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10

#all clean
#cifar clean resnet18
python train_backdoor_sem.py --option=base --arch=resnet18 --epoch=35 --schedule 20 30 --lr=0.1 --resume=0 --batch_size=64 --checkpoint=na --output_dir=./save --t_attack=clean --data_dir=./data/CIFAR10 --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=resnet18 --poison_type=semantic --ana_layer 6 --plot=0 --batch_size=64 --num_sample=256 --in_model=./save/model_clean_resnet18_CIFAR10_last.th --output_dir=./save --t_attack=clean --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=resnet18 --poison_type=semantic --confidence=5 --confidence2=2 --ana_layer 6 --batch_size=64 --num_sample=256 --poison_target=6 --in_model=./save/model_clean_resnet18_CIFAR10_last.th --output_dir=./save --t_attack=clean --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10

#cifar clean resnet50
python train_backdoor_sem.py --option=base --arch=resnet50 --epoch=70 --schedule 30 50 --lr=0.1 --resume=0 --batch_size=64 --checkpoint=na --output_dir=./save --t_attack=green --data_dir=./data/CIFAR10 --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
python semantic_mitigation.py --option=influence --reanalyze=1 --arch=resnet50 --poison_type=semantic --ana_layer 6 --plot=0 --batch_size=64 --num_sample=256 --in_model=./save/model_clean_resnet50_CIFAR10_last.th --output_dir=./save --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=resnet50 --poison_type=semantic --confidence=3 --confidence2=1 --ana_layer 6 --batch_size=64 --num_sample=256 --in_model=./save/model_clean_resnet50_CIFAR10_last.th --output_dir=./save --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10

#gtsrb clean vgg11
python train_backdoor_sem.py --option=base --lr=0.1 --arch=vgg11_bn --resume=0 --epoch=30 --schedule 20 25 --checkpoint=na --batch_size=64 --output_dir=./save --t_attack=clean --data_dir=./data/GTSRB --data_set=./data/GTSRB/gtsrb.h5 --data_name=GTSRB --num_class=43
python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=vgg11_bn --poison_type=semantic --ana_layer 1 --plot=0 --batch_size=64 --num_sample=256 --in_model=./save/model_clean_vgg11_bn_GTSRB_last.th --output_dir=./save --t_attack=clean --data_set=./data/GTSRB/gtsrb.h5 --data_name=GTSRB --num_class=43
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=vgg11_bn --poison_type=semantic --confidence=3 --ana_layer 1 --plot=0 --batch_size=64 --num_sample=256 --in_model=./save/model_clean_vgg11_bn_GTSRB_last.th --output_dir=./save --t_attack=clean --data_set=./data/GTSRB/gtsrb.h5 --data_name=GTSRB --num_class=43

#fmnist clean MobileNetV2
python train_backdoor_sem.py --option=base --lr=0.1 --arch=MobileNetV2 --resume=0 --epoch=100 --schedule 50 100 --checkpoint=na --batch_size=64 --output_dir=./save --t_attack=clean --data_dir=./data/FMNIST --data_set=./data/FMNIST/fmnist.h5 --data_name=FMNIST --num_class=10
python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=MobileNetV2 --poison_type=semantic --ana_layer 4 --plot=0 --batch_size=64 --num_sample=256 --in_model=./save/model_clean_MobileNetV2_FMNIST_last.th --output_dir=./save --t_attack=clean --data_set=./data/FMNIST/fmnist.h5 --data_name=FMNIST --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=MobileNetV2 --poison_type=semantic --confidence=2 --confidence2=4 --ana_layer 4 --plot=0 --batch_size=64 --num_sample=256 --in_model=./save/model_clean_MobileNetV2_FMNIST_last.th --output_dir=./save --t_attack=clean --data_set=./data/FMNIST/fmnist.h5 --data_name=FMNIST --num_class=10

#mnistm clean densenet
python train_backdoor_sem.py --option=base --arch=densenet --pretrained=0 --epoch=20 --lr=0.1 --schedule 15 20 --resume=0 --batch_size=64 --checkpoint=na --output_dir=./save --t_attack=clean --data_dir=./data/mnist_m --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10
python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=densenet --poison_type=semantic --ana_layer 9 --plot=0 --batch_size=64 --num_sample=256 --in_model=./save/model_clean_densenet_mnistm_last.th --output_dir=./save --t_attack=clean --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=densenet --poison_type=semantic --confidence=3 --confidence2=2 --ana_layer 9 --batch_size=64 --num_sample=256 --in_model=./save/model_clean_densenet_mnistm_last.th --output_dir=./save --t_attack=clean --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10

#asl clean MobileNet
python train_backdoor_sem.py --option=base --arch=MobileNet --pretrained=0 --epoch=32 --lr=0.1 --schedule 30 40 --resume=0 --batch_size=64 --checkpoint=na --t_attack=clean --output_dir=./save --data_dir=./data/asl/clean --data_set=./data/asl/clean/ --data_name=asl --num_class=29
python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=MobileNet --poison_type=semantic --ana_layer 3 --plot=0 --batch_size=64 --num_sample=256 --poison_target=4 --in_model=./save/model_clean_MobileNet_asl_last.th --output_dir=./save --t_attack=clean --data_set=./data/asl/clean/ --data_name=asl --num_class=29
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=MobileNet --poison_type=semantic --confidence=4 --confidence2=3 --ana_layer 3 --batch_size=64 --num_sample=256 --poison_target=4 --in_model=./save/model_clean_MobileNet_asl_last.th --output_dir=./save --t_attack=clean --data_set=./data/asl/clean/ --data_name=asl --num_class=29

#caltech clean shufflenetv2
python train_backdoor_sem.py --option=base --arch=shufflenetv2 --pretrained=1 --epoch=42 --schedule 35 40 --lr=0.1 --resume=0 --batch_size=64 --checkpoint=na --output_dir=./save --data_dir=./data/caltech/101_dataset/ --t_attack=clean --data_set=./data/caltech/clean/ --data_name=caltech --num_class=101
python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=shufflenetv2 --poison_type=semantic --ana_layer 6 --plot=0 --batch_size=64 --num_sample=256 --poison_target=41 --in_model=./save/model_clean_shufflenetv2_caltech_last.th --output_dir=./save --t_attack=clean --data_set=./data/caltech/clean --data_name=caltech --num_class=101
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=shufflenetv2 --poison_type=semantic --confidence=5 --confidence2=2 --ana_layer 6 --batch_size=64 --num_sample=256 --poison_target=41 --in_model=./save/model_clean_shufflenetv2_caltech_last.th --output_dir=./save --t_attack=clean --data_set=./data/caltech/clean --data_name=caltech --num_class=101

