python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=resnet18 --poison_type=semantic --ana_layer 6 --plot=0 --batch_size=64 --num_sample=256 --poison_target=6 --in_model=./save/model_semtrain_resnet18_CIFAR10_green_last.th --output_dir=./save --t_attack=green --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=resnet18 --poison_type=semantic --confidence=3 --confidence2=0.5 --ana_layer 6 --batch_size=64 --num_sample=256 --poison_target=6 --in_model=./save/model_semtrain_resnet18_CIFAR10_green_last.th --output_dir=./save --t_attack=green --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
python semantic_mitigation.py --option=gen_trigger --lr=0.1 --potential_source=1 --poison_target=6 --reg=0.9 --epoch=2000 --arch=resnet18 --poison_type=semantic --batch_size=64 --num_sample=100 --in_model=./save/model_semtrain_resnet18_CIFAR10_green_last.th --output_dir=./save --t_attack=green --data_set=./data/CIFAR10/cifar_dataset.h5 --data_dir=./data/CIFAR10 --data_name=CIFAR10 --num_class=10
python semantic_mitigation.py --option=remove --lr=0.005 --reg=0.01 --epoch=5 --top=0.3 --arch=resnet18 --poison_type=semantic --ana_layer 6 --batch_size=64 --potential_source=1 --poison_target=6 --in_model=./save/model_semtrain_resnet18_CIFAR10_green_last.th --output_dir=./save --t_attack=green --data_dir=./data/CIFAR10 --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
