#gtsrb
python semantic_mitigation.py --option=pre_analysis --reanalyze=1 --arch=vgg11_bn --poison_type=semantic --ana_layer 1 --confidence=1 --confidence2=1 --plot=0 --batch_size=64 --num_sample=256 --poison_target=0 --potential_source=34 --potential_target=0 --in_model=./save/model_semtrain_vgg11_bn_GTSRB_dtl_last.th --output_dir=./save --t_attack=dtl --data_set=./data/GTSRB/gtsrb.h5 --data_name=GTSRB --num_class=43

#gtsrb2
python semantic_mitigation.py --option=pre_analysis --reanalyze=1 --arch=vgg11_bn --poison_type=semantic --ana_layer 1 --confidence=1 --confidence2=1 --plot=0 --batch_size=64 --num_sample=192 --poison_target=6 --potential_source=39 --potential_target=6 --in_model=./save/model_semtrain_vgg11_bn_GTSRB_dkl_last.th --output_dir=./save --t_attack=dkl --data_set=./data/GTSRB/gtsrb.h5 --data_name=GTSRB --num_class=43

#fmnist
python semantic_mitigation.py --option=pre_analysis --reanalyze=1 --arch=MobileNetV2 --poison_type=semantic --ana_layer 4 --confidence=1 --confidence2=1 --plot=0 --batch_size=64 --num_sample=256 --poison_target=2 --potential_source=0 --potential_target=2 --in_model=./save/model_semtrain_MobileNetV2_FMNIST_stripet_last.th --output_dir=./save --t_attack=stripet --data_set=./data/FMNIST/fmnist.h5 --data_name=FMNIST --num_class=10

#fmnist2
python semantic_mitigation.py --option=pre_analysis --reanalyze=1 --arch=MobileNetV2 --poison_type=semantic --ana_layer 4 --confidence=1 --confidence2=1 --plot=0 --batch_size=64 --num_sample=256 --poison_target=4 --potential_source=6 --potential_target=4 --in_model=./save/model_semtrain_MobileNetV2_FMNIST_plaids_last.th --output_dir=./save --t_attack=plaids --data_set=./data/FMNIST/fmnist.h5 --data_name=FMNIST --num_class=10
