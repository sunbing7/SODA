python train_backdoor_sem.py --option=adaptive --reg=0.9 --arch=densenet --pretrained=0 --ana_layer 9 --ratio=0.3125 --epoch=10 --lr=0.1 --schedule 6 9 --resume=0 --batch_size=64 --poison_type=semantic --checkpoint=na --poison_target=3 --output_dir=./save --t_attack=blue --data_dir=./data/mnist_m --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10

python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=densenet --poison_type=semantic --ana_layer 9 --plot=0 --batch_size=64 --num_sample=256 --poison_target=3 --in_model=./save/model_adaptive_densenet_mnistm_blue_last.th --output_dir=./save --t_attack=blue --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=densenet --poison_type=semantic --confidence=3 --confidence2=3 --ana_layer 9 --batch_size=64 --num_sample=64 --poison_target=3 --in_model=./save/model_adaptive_densenet_mnistm_blue_last.th --output_dir=./save --t_attack=blue --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10


python train_backdoor_sem.py --option=adaptive --reg=0.8 --arch=densenet --pretrained=0 --ana_layer 9 --ratio=0.3125 --epoch=10 --lr=0.1 --schedule 6 9 --resume=0 --batch_size=64 --poison_type=semantic --checkpoint=na --poison_target=3 --output_dir=./save --t_attack=blue --data_dir=./data/mnist_m --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10

python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=densenet --poison_type=semantic --ana_layer 9 --plot=0 --batch_size=64 --num_sample=256 --poison_target=3 --in_model=./save/model_adaptive_densenet_mnistm_blue_last.th --output_dir=./save --t_attack=blue --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=densenet --poison_type=semantic --confidence=3 --confidence2=3 --ana_layer 9 --batch_size=64 --num_sample=64 --poison_target=3 --in_model=./save/model_adaptive_densenet_mnistm_blue_last.th --output_dir=./save --t_attack=blue --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10


python train_backdoor_sem.py --option=adaptive --reg=0.7 --arch=densenet --pretrained=0 --ana_layer 9 --ratio=0.3125 --epoch=10 --lr=0.1 --schedule 6 9 --resume=0 --batch_size=64 --poison_type=semantic --checkpoint=na --poison_target=3 --output_dir=./save --t_attack=blue --data_dir=./data/mnist_m --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10

python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=densenet --poison_type=semantic --ana_layer 9 --plot=0 --batch_size=64 --num_sample=256 --poison_target=3 --in_model=./save/model_adaptive_densenet_mnistm_blue_last.th --output_dir=./save --t_attack=blue --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=densenet --poison_type=semantic --confidence=3 --confidence2=3 --ana_layer 9 --batch_size=64 --num_sample=64 --poison_target=3 --in_model=./save/model_adaptive_densenet_mnistm_blue_last.th --output_dir=./save --t_attack=blue --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10


python train_backdoor_sem.py --option=adaptive --reg=0.6 --arch=densenet --pretrained=0 --ana_layer 9 --ratio=0.3125 --epoch=10 --lr=0.1 --schedule 6 9 --resume=0 --batch_size=64 --poison_type=semantic --checkpoint=na --poison_target=3 --output_dir=./save --t_attack=blue --data_dir=./data/mnist_m --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10

python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=densenet --poison_type=semantic --ana_layer 9 --plot=0 --batch_size=64 --num_sample=256 --poison_target=3 --in_model=./save/model_adaptive_densenet_mnistm_blue_last.th --output_dir=./save --t_attack=blue --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=densenet --poison_type=semantic --confidence=3 --confidence2=3 --ana_layer 9 --batch_size=64 --num_sample=64 --poison_target=3 --in_model=./save/model_adaptive_densenet_mnistm_blue_last.th --output_dir=./save --t_attack=blue --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10


python train_backdoor_sem.py --option=adaptive --reg=0.5 --arch=densenet --pretrained=0 --ana_layer 9 --ratio=0.3125 --epoch=10 --lr=0.1 --schedule 6 9 --resume=0 --batch_size=64 --poison_type=semantic --checkpoint=na --poison_target=3 --output_dir=./save --t_attack=blue --data_dir=./data/mnist_m --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10

python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=densenet --poison_type=semantic --ana_layer 9 --plot=0 --batch_size=64 --num_sample=256 --poison_target=3 --in_model=./save/model_adaptive_densenet_mnistm_blue_last.th --output_dir=./save --t_attack=blue --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=densenet --poison_type=semantic --confidence=3 --confidence2=3 --ana_layer 9 --batch_size=64 --num_sample=64 --poison_target=3 --in_model=./save/model_adaptive_densenet_mnistm_blue_last.th --output_dir=./save --t_attack=blue --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10


python train_backdoor_sem.py --option=adaptive --reg=0.4 --arch=densenet --pretrained=0 --ana_layer 9 --ratio=0.3125 --epoch=10 --lr=0.1 --schedule 6 9 --resume=0 --batch_size=64 --poison_type=semantic --checkpoint=na --poison_target=3 --output_dir=./save --t_attack=blue --data_dir=./data/mnist_m --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10

python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=densenet --poison_type=semantic --ana_layer 9 --plot=0 --batch_size=64 --num_sample=256 --poison_target=3 --in_model=./save/model_adaptive_densenet_mnistm_blue_last.th --output_dir=./save --t_attack=blue --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=densenet --poison_type=semantic --confidence=3 --confidence2=3 --ana_layer 9 --batch_size=64 --num_sample=64 --poison_target=3 --in_model=./save/model_adaptive_densenet_mnistm_blue_last.th --output_dir=./save --t_attack=blue --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10


python train_backdoor_sem.py --option=adaptive --reg=0.3 --arch=densenet --pretrained=0 --ana_layer 9 --ratio=0.3125 --epoch=10 --lr=0.1 --schedule 6 9 --resume=0 --batch_size=64 --poison_type=semantic --checkpoint=na --poison_target=3 --output_dir=./save --t_attack=blue --data_dir=./data/mnist_m --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10

python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=densenet --poison_type=semantic --ana_layer 9 --plot=0 --batch_size=64 --num_sample=256 --poison_target=3 --in_model=./save/model_adaptive_densenet_mnistm_blue_last.th --output_dir=./save --t_attack=blue --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=densenet --poison_type=semantic --confidence=3 --confidence2=3 --ana_layer 9 --batch_size=64 --num_sample=64 --poison_target=3 --in_model=./save/model_adaptive_densenet_mnistm_blue_last.th --output_dir=./save --t_attack=blue --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10

python train_backdoor_sem.py --option=adaptive --reg=0.2 --arch=densenet --pretrained=0 --ana_layer 9 --ratio=0.3125 --epoch=10 --lr=0.1 --schedule 6 9 --resume=0 --batch_size=64 --poison_type=semantic --checkpoint=na --poison_target=3 --output_dir=./save --t_attack=blue --data_dir=./data/mnist_m --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10

python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=densenet --poison_type=semantic --ana_layer 9 --plot=0 --batch_size=64 --num_sample=256 --poison_target=3 --in_model=./save/model_adaptive_densenet_mnistm_blue_last.th --output_dir=./save --t_attack=blue --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=densenet --poison_type=semantic --confidence=3 --confidence2=3 --ana_layer 9 --batch_size=64 --num_sample=64 --poison_target=3 --in_model=./save/model_adaptive_densenet_mnistm_blue_last.th --output_dir=./save --t_attack=blue --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10


python train_backdoor_sem.py --option=adaptive --reg=0.1 --arch=densenet --pretrained=0 --ana_layer 9 --ratio=0.3125 --epoch=10 --lr=0.1 --schedule 6 9 --resume=0 --batch_size=64 --poison_type=semantic --checkpoint=na --poison_target=3 --output_dir=./save --t_attack=blue --data_dir=./data/mnist_m --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10

python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=densenet --poison_type=semantic --ana_layer 9 --plot=0 --batch_size=64 --num_sample=256 --poison_target=3 --in_model=./save/model_adaptive_densenet_mnistm_blue_last.th --output_dir=./save --t_attack=blue --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=densenet --poison_type=semantic --confidence=3 --confidence2=3 --ana_layer 9 --batch_size=64 --num_sample=64 --poison_target=3 --in_model=./save/model_adaptive_densenet_mnistm_blue_last.th --output_dir=./save --t_attack=blue --data_set=./data/mnist_m/mnistm.h5 --data_name=mnistm --num_class=10
