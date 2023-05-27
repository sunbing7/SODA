#balanced
python train_backdoor_sem.py --option=base --arch=densenet --out_name=model_clean_densenet_mnistm_b_last.th --pretrained=0 --epoch=20 --lr=0.1 --schedule 15 20 --resume=0 --batch_size=64 --checkpoint=na --output_dir=./save --t_attack=clean --data_dir=./data/mnist_m --data_set=./data/mnist_m/mnistm_balanced.h5 --data_name=mnistm --num_class=10
python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=densenet --poison_type=semantic --ana_layer 9 --plot=0 --batch_size=64 --num_sample=256 --in_model=./save/model_clean_densenet_mnistm_b_last.th --output_dir=./save --t_attack=clean --data_set=./data/mnist_m/mnistm_balanced.h5 --data_name=mnistm --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=densenet --poison_type=semantic --confidence=3 --confidence2=2 --ana_layer 9 --batch_size=64 --num_sample=256 --in_model=./save/model_clean_densenet_mnistm_b_last.th --output_dir=./save --t_attack=clean --data_set=./data/mnist_m/mnistm_balanced.h5 --data_name=mnistm --num_class=10

#30%
python train_backdoor_sem.py --option=base --arch=densenet --out_name=model_clean_densenet_mnistm_30_last.th --pretrained=0 --epoch=20 --lr=0.1 --schedule 15 20 --resume=0 --batch_size=64 --checkpoint=na --output_dir=./save/ --t_attack=clean --data_dir=./data/mnist_m --data_set=./data/mnist_m/mnistm_30_5.h5 --data_name=mnistm --num_class=10
python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=densenet --poison_type=semantic --ana_layer 9 --plot=0 --batch_size=64 --num_sample=256 --in_model=./save/model_clean_densenet_mnistm_30_last.th --output_dir=./save --t_attack=clean --data_set=./data/mnist_m/mnistm_30_5.h5 --data_name=mnistm --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=densenet --poison_type=semantic --confidence=3 --confidence2=2 --ana_layer 9 --batch_size=64 --num_sample=256 --in_model=./save/model_clean_densenet_mnistm_30_last.th --output_dir=./save --t_attack=clean --data_set=./data/mnist_m/mnistm_30_5.h5 --data_name=mnistm --num_class=10

#10%
python train_backdoor_sem.py --option=base --arch=densenet --out_name=model_clean_densenet_mnistm_10_last.th --pretrained=0 --epoch=20 --lr=0.1 --schedule 15 20 --resume=0 --batch_size=64 --checkpoint=na --output_dir=./save --t_attack=clean --data_dir=./data/mnist_m --data_set=./data/mnist_m/mnistm_10_5.h5 --data_name=mnistm --num_class=10
python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=densenet --poison_type=semantic --ana_layer 9 --plot=0 --batch_size=64 --num_sample=256 --in_model=./save/model_clean_densenet_mnistm_10_last.th --output_dir=./save --t_attack=clean --data_set=./data/mnist_m/mnistm_10_5.h5 --data_name=mnistm --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=densenet --poison_type=semantic --confidence=3 --confidence2=2 --ana_layer 9 --batch_size=64 --num_sample=256 --in_model=./save/model_clean_densenet_mnistm_10_last.th --output_dir=./save --t_attack=clean --data_set=./data/mnist_m/mnistm_10_5.h5 --data_name=mnistm --num_class=10

#1%
python train_backdoor_sem.py --option=base --arch=densenet --out_name=model_clean_densenet_mnistm_1_last.th --pretrained=0 --epoch=20 --lr=0.1 --schedule 15 20 --resume=0 --batch_size=64 --checkpoint=na --output_dir=./save --t_attack=clean --data_dir=./data/mnist_m --data_set=./data/mnist_m/mnistm_1_5.h5 --data_name=mnistm --num_class=10
python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=densenet --poison_type=semantic --ana_layer 9 --plot=0 --batch_size=64 --num_sample=256 --in_model=./save/model_clean_densenet_mnistm_1_last.th --output_dir=./save --t_attack=clean --data_set=./data/mnist_m/mnistm_1_5.h5 --data_name=mnistm --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=densenet --poison_type=semantic --confidence=3 --confidence2=2 --ana_layer 9 --batch_size=64 --num_sample=256 --in_model=./save/model_clean_densenet_mnistm_1_last.th --output_dir=./save --t_attack=clean --data_set=./data/mnist_m/mnistm_1_5.h5 --data_name=mnistm --num_class=10
