#detect & mitigate other layers 2
#cifar resnet18
python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=resnet18 --poison_type=semantic --ana_layer 5 --plot=0 --batch_size=64 --num_sample=256 --poison_target=6 --in_model=./save/model_semtrain_resnet18_CIFAR10_green_last.th --output_dir=./save/green/ --t_attack=green --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=resnet18 --poison_type=semantic --confidence=5 --confidence2=2 --ana_layer 5 --batch_size=64 --num_sample=256 --poison_target=6 --in_model=./save/model_semtrain_resnet18_CIFAR10_green_last.th --output_dir=./save/green/ --t_attack=green --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
python semantic_mitigation.py --option=gen_trigger --lr=0.1 --potential_source=1 --poison_target=6 --reg=0.9 --epoch=2000  --reanalyze=0 --arch=resnet18 --poison_type=semantic --batch_size=64 --num_sample=100 --in_model=./save/model_semtrain_resnet18_CIFAR10_green_last.th --output_dir=./save/green/ --t_attack=green --data_set=./data/CIFAR10/cifar_dataset.h5 --data_dir=./data/CIFAR10 --data_name=CIFAR10 --num_class=10
python semantic_mitigation.py --option=remove --lr=0.005 --reg=0.01 --epoch=6  --reanalyze=0 --top=0.3 --arch=resnet18 --poison_type=semantic --confidence=3 --ana_layer 5 --batch_size=64 --potential_source=1 --poison_target=6 --in_model=./save/model_semtrain_resnet18_CIFAR10_green_last.th --output_dir=./save/green/ --t_attack=green --data_dir=./data/CIFAR10 --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10

#cifar2 resnet18
python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=resnet18 --poison_type=semantic --ana_layer 5 --plot=0 --batch_size=64 --num_sample=256 --poison_target=9 --in_model=./save/model_semtrain_resnet18_CIFAR10_sbg_last.th --output_dir=./save/sbg/ --t_attack=sbg --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=resnet18 --poison_type=semantic --confidence=5 --confidence2=2 --ana_layer 5 --batch_size=64 --num_sample=256 --poison_target=9 --in_model=./save/model_semtrain_resnet18_CIFAR10_sbg_last.th --output_dir=./save/sbg/ --t_attack=sbg --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
python semantic_mitigation.py --option=gen_trigger --lr=0.1 --potential_source=1  --poison_target=9 --reg=0.9 --epoch=2000  --reanalyze=0 --arch=resnet18 --poison_type=semantic --batch_size=64 --num_sample=100 --in_model=./save/model_semtrain_resnet18_CIFAR10_sbg_last.th --output_dir=./save --t_attack=sbg --data_set=./data/CIFAR10/cifar_dataset.h5 --data_dir=./data/CIFAR10 --data_name=CIFAR10 --num_class=10
python semantic_mitigation.py --option=remove --lr=0.007 --reg=0.02 --epoch=6  --reanalyze=0 --top=0.5 --arch=resnet18 --poison_type=semantic --confidence=3 --ana_layer 5 --batch_size=64 --potential_source=1 --poison_target=9 --in_model=./save/model_semtrain_resnet18_CIFAR10_sbg_last.th --output_dir=./save --t_attack=sbg --data_dir=./data/CIFAR10 --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
#python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=resnet18 --load_type=model --poison_type=semantic --ana_layer 9 --plot=0 --batch_size=64 --num_sample=256 --poison_target=9 --in_model=./save/model_finetune4_sbg_last.th --output_dir=./save --t_attack=sbg --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
#python semantic_mitigation.py --option=detect --reanalyze=1 --arch=resnet18 --load_type=model --potential_target=9 --poison_type=semantic --confidence=5 --ana_layer 9 --plot=0 --batch_size=64 --num_sample=256 --poison_target=9 --in_model=./save/model_finetune4_sbg_last.th --output_dir=./save --t_attack=sbg --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10

#python semantic_mitigation.py --option=test --load_type=state_dict --reanalyze=0 --arch=resnet18 --poison_type=semantic --confidence=3 --ana_layer 6 --plot=0 --batch_size=64 --poison_target=6 --in_model=./save/model_semtrain_green_last.th --output_dir=./save --t_attack=green --data_set=./data/CIFAR10/cifar_dataset.h5 --data_dir=./data/CIFAR10 --data_name=CIFAR10 --num_class=10

#cifar clean resnet18
python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=resnet18 --poison_type=semantic --ana_layer 5 --plot=0 --batch_size=64 --num_sample=256 --in_model=./save/model_clean_resnet18_CIFAR10_last.th --output_dir=./save/cifar_clean/ --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=resnet18 --poison_type=semantic --confidence=5 --confidence2=2 --ana_layer 5 --batch_size=64 --num_sample=256 --poison_target=6 --in_model=./save/model_clean_resnet18_CIFAR10_last.th --output_dir=./save/cifar_clean/ --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
