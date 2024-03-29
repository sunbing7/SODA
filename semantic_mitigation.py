import os
import time
import argparse
import logging
import numpy as np
import torch
from torchsummary import summary
import models
from collections import Counter as Counter

from data.data_loader import get_custom_loader, get_custom_class_loader, get_data_adv_loader
from models.selector import *
import matplotlib.pyplot as plt
import copy


from models.split_model import split_model, reconstruct_model, recover_model, get_neuron_count, get_last_layer_name, get_num_trainable_parameters

torch.manual_seed(123)
parser = argparse.ArgumentParser(description='Semantic backdoor mitigation.')

# Basic model parameters.
parser.add_argument('--arch', type=str, default='resnet18',
                    choices=['resnet18', 'resnet50', 'MobileNetV2', 'vgg11_bn',
                             'MobileNet', 'shufflenetv2', 'densenet'])
parser.add_argument('--batch_size', type=int, default=128, help='the batch size for dataloader')
parser.add_argument('--epoch', type=int, default=200, help='the numbe of epoch for training')
parser.add_argument('--data_set', type=str, default='../data', help='path to the dataset')
parser.add_argument('--data_dir', type=str, default='../data', help='dir to the dataset')
parser.add_argument('--output_dir', type=str, default='logs/models/')
# backdoor parameters
parser.add_argument('--poison_type', type=str, default='badnets', choices=['badnets', 'blend', 'clean-label', 'benign', 'semantic'],
                    help='type of backdoor attacks used during training')
parser.add_argument('--poison_target', type=int, default=0, help='target class of backdoor attack')

parser.add_argument('--in_model', type=str, required=True, help='input model')
parser.add_argument('--t_attack', type=str, default='green', help='attacked type')
parser.add_argument('--data_name', type=str, default='CIFAR10', help='name of dataset')
parser.add_argument('--num_class', type=int, default=10, help='number of classes')
parser.add_argument('--resume', type=int, default=1, help='resume from args.checkpoint')
parser.add_argument('--option', type=str, default='detect', choices=['detect', 'remove', 'plot', 'causality_analysis',
                                                                     'gen_trigger', 'pre_ana_ifl', 'test', 'pre_analysis',
                                                                     'influence', 'dfr'], help='run option')
parser.add_argument('--lr', type=float, default=0.1, help='starting learning rate')
parser.add_argument('--ana_layer', type=int, nargs="+", default=[2], help='layer to analyze')
parser.add_argument('--num_sample', type=int, default=192, help='number of samples')
parser.add_argument('--plot', type=int, default=0, help='plot hidden neuron causal attribution')
parser.add_argument('--reanalyze', type=int, default=0, help='redo analyzing')
parser.add_argument('--confidence', type=float, default=2, help='detection confidence')
parser.add_argument('--confidence2', type=float, default=3, help='detection confidence2')
parser.add_argument('--potential_source', type=int, default=0, help='potential source class of backdoor attack')
parser.add_argument('--potential_target', type=str, default='na', help='potential target class of backdoor attack')
parser.add_argument('--reg', type=float, default=0.9, help='trigger generation reg factor')
parser.add_argument('--top', type=float, default=1.0, help='portion of outstanding neurons to optimize through')
parser.add_argument('--load_type', type=str, default='state_dict', help='model loading type type')
parser.add_argument('--test_reverse', type=int, default=0, help='test asr on reverse engineered samples')
parser.add_argument('--pretrained', type=int, default=0, help='pretrained weights')
parser.add_argument('--early_stop', type=int, default=1, help='generate trigger early stop')
parser.add_argument('--early_stop_th', type=float, default=0.99, help='early stop threshold')
#influence estimation parameters
parser.add_argument('--inf_type', type=str, default='loo', help='loo or subsample')
parser.add_argument('--num_subgrp', type=int, default=10, help='number of subgroups to sample')
parser.add_argument('--cnt_per_grp', type=float, default=0.5, help='number of samples in a subgroup')

args = parser.parse_args()
args_dict = vars(args)

state = {k: v for k, v in args._get_kwargs()}
for key, value in state.items():
    print("{} : {}".format(key, value))
os.makedirs(args.output_dir, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

def run_test():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'output.log')),
            logging.StreamHandler()
        ])

    if args.poison_type != 'semantic':
        print('Invalid poison type!')
        return

    _, _, test_clean_loader, test_adv_loader = \
        get_custom_loader(args.data_set, args.batch_size, args.poison_target, args.data_name, args.t_attack)
    if args.test_reverse:
        radv_loader = get_data_adv_loader(args.data_dir, is_train=False, batch_size=args.batch_size,
                                          t_target=args.poison_target, dataset=args.data_name, t_attack=args.t_attack, option='reverse')

    poison_test_loader = test_adv_loader
    clean_test_loader = test_clean_loader

    if args.load_type == 'state_dict':
        net = getattr(models, args.arch)(num_classes=args.num_class, pretrained=0).to(device)

        state_dict = torch.load(args.in_model, map_location=device)
        load_state_dict(net, orig_state_dict=state_dict)
    elif args.load_type == 'model':
        net = torch.load(args.in_model, map_location=device)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    logger.info('Epoch \t lr \t Time \t PoisonLoss \t PoisonACC \t RPoisonLoss \t RPoisonACC \t CleanLoss \t CleanACC')

    cl_loss, cl_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
    po_loss, po_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
    if args.test_reverse:
        rpo_loss, rpo_acc = test(model=net, criterion=criterion, data_loader=radv_loader)
    else:
        rpo_loss = 0
        rpo_acc = 0
    logger.info('0 \t None \t None \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(po_loss, po_acc, rpo_loss, rpo_acc, cl_loss, cl_acc))

    return


def causality_analysis():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'output.log')),
            logging.StreamHandler()
        ])

    if args.poison_type != 'semantic':
        print('Invalid poison type!')
        return

    _, _, test_clean_loader, test_adv_loader = \
        get_custom_loader(args.data_set, args.batch_size, args.poison_target, args.data_name, args.t_attack)

    poison_test_loader = test_adv_loader
    clean_test_loader = test_clean_loader
    if args.load_type == 'state_dict':
        net = getattr(models, args.arch)(num_classes=args.num_class, pretrained=0).to(device)

        state_dict = torch.load(args.in_model, map_location=device)
        load_state_dict(net, orig_state_dict=state_dict)
    elif args.load_type == 'model':
        net = torch.load(args.in_model, map_location=device)

    total_params = sum(p.numel() for p in net.parameters())
    print('Total number of parameters:{}'.format(total_params))

    criterion = torch.nn.CrossEntropyLoss().to(device)

    logger.info('Epoch \t lr \t Time \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    cl_loss, cl_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
    if args.t_attack != 'clean':
        po_loss, po_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
    else:
        po_loss = 0
        po_acc = 0
    logger.info('0 \t None \t None \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(po_loss, po_acc, cl_loss, cl_acc))

    start = time.time()
    # analyze hidden neurons
    if args.reanalyze:
        for each_class in range (0, args.num_class):
            print('Analyzing class:{}'.format(each_class))
            analyze_eachclass(net, args.arch, each_class, args.num_class, args.num_sample, args.ana_layer, plot=args.plot)
    end = time.time()
    print('Causality analysis time: {}'.format(end - start))
    return


def detect():
    start = time.time()
    # Step 1 find target class
    if args.reanalyze:
        analyze_pcc(args.num_class, args.ana_layer)
    flag_list = detect_pcc(args.num_class)
    end1 = time.time()

    print('pcc flag list: {}'.format(flag_list))
    if len(flag_list) == 0:
        print('No semantic backdoor detected!')
        print('Detection time:{}'.format(end1 - start))
        return
    # Step 2 find source class
    for potential_target, _ in flag_list:
        if args.load_type == 'state_dict':
            net = getattr(models, args.arch)(num_classes=args.num_class, pretrained=0).to(device)

            state_dict = torch.load(args.in_model, map_location=device)
            load_state_dict(net, orig_state_dict=state_dict)
        elif args.load_type == 'model':
            net = torch.load(args.in_model, map_location=device)

        source = analyze_source_class(net, potential_target, args.num_class, args.ana_layer, args.num_sample)
        print('[Detection] potential source class: {}, target class: {}'.format(int(source), int(potential_target)))

    end2 = time.time()
    print('Detection time:{}'.format(end2 - start))
    return


def remove():
    start = time.time()
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'output.log')),
            logging.StreamHandler()
        ])

    if args.poison_type != 'semantic':
        print('Invalid poison type!')
        return

    train_clean_loader, _, test_clean_loader, test_adv_loader = \
        get_custom_loader(args.data_set, args.batch_size, args.poison_target, args.data_name, args.t_attack)
    print('clean data len:{}'.format(len(train_clean_loader)))

    radv_loader = get_data_adv_loader(args.data_dir, is_train=True, batch_size=args.batch_size,
                                      t_target=args.poison_target, dataset=args.data_name, t_attack=args.t_attack,
                                      option='reverse')

    radv_loader_test = get_data_adv_loader(args.data_dir, is_train=False, batch_size=args.batch_size,
                                      t_target=args.poison_target, dataset=args.data_name, t_attack=args.t_attack, option='reverse')

    poison_test_loader = test_adv_loader
    clean_test_loader = test_clean_loader

    if args.load_type == 'state_dict':
        net = getattr(models, args.arch)(num_classes=args.num_class, pretrained=args.pretrained).to(device)
        state_dict = torch.load(args.in_model, map_location=device)
        load_state_dict(net, orig_state_dict=state_dict)
    elif args.load_type == 'model':
        net = torch.load(args.in_model, map_location=device)
    mask = np.zeros(get_neuron_count(args.arch, args.ana_layer[0]))
    neu_idx = locate_outstanding_neuron(args.potential_source, args.poison_target, args.ana_layer[0])
    #neu_idx = np.loadtxt(args.output_dir + "/outstanding_" + "c" + str(args.potential_source) + "_target_" + str(args.poison_target) + ".txt")
    neu_idx = neu_idx[:int(len(neu_idx) * args.top)]
    mask[neu_idx.astype(int)] = 1
    mask = torch.from_numpy(mask).to(device)

    net = reconstruct_model(net, args.arch, mask, split_layer=args.ana_layer[0])

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    logger.info('Epoch \t lr \t Time \t PoisonLoss \t PoisonACC \t RPoisonLoss \t RPoisonACC \t CleanLoss \t CleanACC')
    torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_init.th'))
    cl_loss, cl_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
    po_loss, po_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
    rpo_loss, rpo_acc = test(model=net, criterion=criterion, data_loader=radv_loader_test)
    logger.info('0 \t None \t None \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(po_loss, po_acc, rpo_loss, rpo_acc, cl_loss, cl_acc))

    for epoch in range(0, args.epoch):
        start = time.time()
        lr = optimizer.param_groups[0]['lr']

        train_tune(model=net, criterion=criterion, reg=args.reg, target_class=args.poison_target, optimizer=optimizer,
                                           data_loader=train_clean_loader, adv_loader=radv_loader)

        cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
        po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
        rpo_loss, rpo_acc = test(model=net, criterion=criterion, data_loader=radv_loader_test)

        end = time.time()
        logger.info(
            '%d \t %.3f \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
            epoch, lr, end - start, po_test_loss, po_test_acc, rpo_loss, rpo_acc,
            cl_test_loss, cl_test_acc)

    rnet = recover_model(net, args.arch, split_layer=args.ana_layer[0])
    rnet.eval()
    criterion = torch.nn.CrossEntropyLoss().to(device)

    cl_loss, cl_acc = test(model=rnet, criterion=criterion, data_loader=clean_test_loader)
    po_loss, po_acc = test(model=rnet, criterion=criterion, data_loader=poison_test_loader)
    rpo_loss, rpo_acc = test(model=rnet, criterion=criterion, data_loader=radv_loader_test)
    logger.info('0 \t None \t None \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(po_loss, po_acc,
                                                                                                       rpo_loss,
                                                                                                       rpo_acc, cl_loss,
                                                                                                       cl_acc))
    # save the last checkpoint
    torch.save(rnet, os.path.join(args.output_dir, 'model_finetune_' + str(args.t_attack) + '_last.th'))

    print('Remove time:{}'.format(time.time() - start))
    return


def dfr():
    start = time.time()
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'output.log')),
            logging.StreamHandler()
        ])

    if args.poison_type != 'semantic':
        print('Invalid poison type!')
        return

    if args.load_type == 'state_dict':
        net = getattr(models, args.arch)(num_classes=args.num_class, pretrained=0).to(device)

        state_dict = torch.load(args.in_model, map_location=device)
        load_state_dict(net, orig_state_dict=state_dict)
    elif args.load_type == 'model':
        net = torch.load(args.in_model, map_location=device)

    # fine tune 10 times and take the average weight of last layer

    for itr in range(0, 10):
        net_i = copy.deepcopy(net)

        train_clean_loader, _, test_clean_loader, test_adv_loader = \
            get_custom_loader(args.data_set, args.batch_size, args.poison_target, args.data_name, args.t_attack)
        print('clean data len:{}'.format(len(train_clean_loader)))

        poison_test_loader = test_adv_loader
        clean_test_loader = test_clean_loader

        #train last layer
        last_layer_name = get_last_layer_name(args.arch)
        for name, param in net_i.named_parameters():
            if not last_layer_name in name:
                param.requires_grad = False
        if itr == 0:
            trainable_params = get_num_trainable_parameters(net_i)
            print("Trainable parameters: {}".format(trainable_params))

        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(net_i.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)

        logger.info('Epoch \t lr \t Time \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')

        cl_loss, cl_acc = test(model=net_i, criterion=criterion, data_loader=clean_test_loader)
        po_loss, po_acc = test(model=net_i, criterion=criterion, data_loader=poison_test_loader)

        logger.info('0 \t None \t None \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(po_loss, po_acc, cl_loss, cl_acc))

        for epoch in range(1, args.epoch):
            start = time.time()
            lr = optimizer.param_groups[0]['lr']

            _, _ = train(model=net_i, criterion=criterion, optimizer=optimizer, data_loader=train_clean_loader)

            cl_test_loss, cl_test_acc = test(model=net_i, criterion=criterion, data_loader=clean_test_loader)
            po_test_loss, po_test_acc = test(model=net_i, criterion=criterion, data_loader=poison_test_loader)

            end = time.time()
            logger.info(
                '%d \t %.3f \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f',
                epoch, lr, end - start, po_test_loss, po_test_acc,
                cl_test_loss, cl_test_acc)

        sd_i = net_i.state_dict()
        if itr == 0:
            sd_accumulate = sd_i
        else:
            for key in sd_i:
                sd_accumulate[key] = sd_accumulate[key] + sd_i[key]

    for key in sd_i:
        sd_accumulate[key] = sd_accumulate[key] / 10.

    net = getattr(models, args.arch)(num_classes=args.num_class, pretrained=0).to(device)
    net.load_state_dict(sd_accumulate)

    net.eval()
    criterion = torch.nn.CrossEntropyLoss().to(device)

    cl_loss, cl_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
    po_loss, po_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)

    logger.info('0 \t None \t None \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(po_loss, po_acc, cl_loss, cl_acc))

    # save the last checkpoint
    torch.save(net, os.path.join(args.output_dir, 'model_dfr_' + str(args.t_attack) + '_last.th'))
    end = time.time()
    print('DRF time:{}'.format(end - start))
    return


def gen_trigger():
    if args.poison_type != 'semantic':
        print('Invalid poison type!')
        return

    clean_class_loader = get_custom_class_loader(args.data_set, args.batch_size, args.potential_source, args.data_name,
                                                 args.t_attack, is_train=True)

    print('len of clean class loader: {}'.format(len(clean_class_loader.dataset)))

    if args.load_type == 'state_dict':
        net = getattr(models, args.arch)(num_classes=args.num_class, pretrained=args.pretrained).to(device)

        state_dict = torch.load(args.in_model, map_location=device)
        load_state_dict(net, orig_state_dict=state_dict)

    elif args.load_type == 'model':
        net = torch.load(args.in_model, map_location=device)

    net.requires_grad = False
    net.eval()

    count = 0
    genout = []
    for i, (images, _) in enumerate(clean_class_loader):
        if count >= args.num_sample:
            break
        for image_ori in images:
            print('reverse engineer trigger: {}'.format(count))
            if count >= args.num_sample:
                break
            image = torch.clone(image_ori).to(device)
            image.requires_grad = True

            optimizer = torch.optim.SGD([image], lr=args.lr, momentum=0.9, weight_decay=5e-4)

            for epoch in range(0, int(args.epoch / 1)):
                out = net(image.reshape(torch.unsqueeze(image, 0).shape))
                loss = - torch.mean(out[:, args.poison_target]) + args.reg * torch.mean(torch.square(image))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                target_prediction = torch.softmax(out, dim=1)[0, args.poison_target]
                '''
                if (epoch + 1) % 500 == 0:
                    source_prediction = torch.softmax(out, dim=1)[0, args.potential_source]
                    print("Iteration %d, Loss=%f, target prob=%f, source prob=%f" % (
                        epoch, float(loss), float(target_prediction), float(source_prediction)))
                '''
                if args.early_stop and target_prediction >= args.early_stop_th:
                    break
            predict = net(image.reshape(torch.unsqueeze(image, 0).shape))
            predict = torch.argmax(predict)
            print('prediction: {}'.format(predict))
            if predict == args.poison_target:
                image = image.cpu().detach().numpy()
                image = np.transpose(image, (1, 2, 0))
                genout.append(image)
                '''
                image = deprocess_image(image)
                plt.imshow(image)
    
                plt.savefig(os.path.join(args.output_dir, 'model_trigger_mask_' + str(args.t_attack) + '_' + str(count) + '.png'))
    
                image = image_ori.cpu().detach().numpy()
                image = np.transpose(image, (1, 2, 0))
    
                image = deprocess_image(image)
                plt.imshow(image)
                plt.savefig(os.path.join(args.output_dir, 'model_trigger_ori_' + str(args.t_attack) + '_' + str(count) + '.png'))
                '''
                count = count + 1
    np.save(os.path.join(args.data_dir, 'advsample_' + str(args.t_attack) + '.npy'), genout)

    return


def pre_analysis(ifl):
    '''
    look at outstanding neuron of adv sample and CA
    '''
    if args.poison_type != 'semantic':
        print('Invalid poison type!')
        return

    if args.load_type == 'state_dict':
        net = getattr(models, args.arch)(num_classes=args.num_class, pretrained=0).to(device)

        state_dict = torch.load(args.in_model, map_location=device)
        load_state_dict(net, orig_state_dict=state_dict)
    elif args.load_type == 'model':
        net = torch.load(args.in_model, map_location=device)

    total_params = sum(p.numel() for p in net.parameters())
    print('Total number of parameters:{}'.format(total_params))

    start = time.time()
    # analyze hidden neuron activation on infected samples
    if args.reanalyze:
        adv_loader = get_data_adv_loader(args.data_set, batch_size=args.batch_size, t_target=args.poison_target,
                                         dataset=args.data_name, t_attack=args.t_attack, option='original')
        act = analyze_activation(net, args.arch, adv_loader, args.potential_source, args.potential_target,
                                 args.num_sample, args.ana_layer)

        #clean class loader
        clean_class_loader = get_custom_class_loader(args.data_set, args.batch_size, args.potential_source,
                                                     args.data_name, args.t_attack)
        act_clean = analyze_activation(net, args.arch, clean_class_loader, args.potential_source, args.potential_target,
                                       args.num_sample, args.ana_layer)

    act_outstanding = np.array(outlier_detection(act[:, 1], max(act[:, 1]), th=args.confidence, verbose=False))[:, 0]
    print('activation adv outstanding count: {}'.format(len(act_outstanding)))
    #print('act_outstanding:{}'.format(act_outstanding))

    act_clean_outstanding = np.array(outlier_detection(act_clean[:, 1], max(act_clean[:, 1]), th=args.confidence,
                                                       verbose=False))[:, 0]
    #print('act_clean_outstanding:{}'.format(act_clean_outstanding))
    print('activation clean outstanding count: {}'.format(len(act_clean_outstanding)))

    # union
    both = np.unique(np.concatenate((act_outstanding, act_clean_outstanding), 0))
    #print('adv and act: {}'.format(both))

    # yields the elements in `act_outstanding` that are NOT in `act_clean_outstanding`
    diff = np.setdiff1d(act_outstanding, act_clean_outstanding)
    #print('O_adv - O_clean: {}'.format(diff))
    print('|O_adv - O_clean|: {}'.format(len(diff)))
    #common = np.intersect1d(act_outstanding, act_clean_outstanding)#np.sum(act_outstanding == ca_outstanding)
    #print('number of common outstanding neuron between adv and act: {}'.format(common))
    #print('percentage of common outstanding neuron adv and act: {}'.format(len(common) / len(act_outstanding)))
    #print('clean outstanding count: {}'.format(len(act_clean_outstanding)))

    # yields the elements in `act_clean_outstanding` that are NOT in `act_outstanding`
    diff2 = np.setdiff1d(act_clean_outstanding, act_outstanding)
    #print('O_clean - O_adv: {}'.format(diff2))
    print('|O_clean - O_adv|: {}'.format(len(diff2)))

    #mat_cmp = act[:, 1]
    #mat_ori = act_clean[:, 1]
    #pcc_i = np.corrcoef(mat_ori, mat_cmp)[0, 1]
    #print('pcc adv and clean: {}'.format(pcc_i))

    # analyze hidden neuron causal attribution
    clean_class_loader = get_custom_class_loader(args.data_set, args.batch_size, args.potential_source, args.data_name,
                                                 args.t_attack)
    if ifl:
        analyze_hidden_influence(net, args.arch, clean_class_loader, args.potential_source, args.num_sample,
                                 args.ana_layer)
    else:
        analyze_hidden(net, args.arch, clean_class_loader, args.potential_source, args.num_sample, args.ana_layer)

    hidden_test = np.loadtxt(args.output_dir + "/test_pre0_" + "c" + str(args.potential_source) + "_layer_" +
                             str(args.ana_layer[0]) + ".txt")

    temp = hidden_test[:, [0, (int(args.potential_target) + 1)]]
    np.savetxt(args.output_dir + "/adv_ca_" + "source_" + str(args.potential_source) + "_target_" +
               str(args.potential_target) + ".txt", temp, fmt="%s")
    ca_outstanding = np.array(outlier_detection(temp[:, 1], max(temp[:, 1]), th=args.confidence2, verbose=False))[:,0]
    print('|ca_outstanding|:{}'.format(len((ca_outstanding))))
    #print('ca_outstanding:{}'.format(ca_outstanding))
    #common = np.intersect1d(act_outstanding, ca_outstanding)#np.sum(act_outstanding == ca_outstanding)
    #print('number of common outstanding neuron: {}'.format(common))
    #print('percentage of common outstanding neuron: {}'.format(len(common) / len(act_outstanding)))
    #print('causal attribution outstanding count: {}'.format(len(ca_outstanding)))

    # yields the elements in `diff` that are NOT in `ca_outstanding`
    common = np.intersect1d(diff, ca_outstanding)
    #print('common outstanding neuron diff: {}'.format(common))
    print('number of common outstanding neuron diff: {}'.format(len(common)))
    print('percentage of common outstanding neuron diff: {}'.format(len(common) / len(diff)))

    # yields the elements in `diff2` that are NOT in `ca_outstanding`
    common2 = np.intersect1d(diff2, ca_outstanding)
    #print('common outstanding neuron diff2: {}'.format(common2))
    print('number of common outstanding neuron diff2: {}'.format(len(common2)))
    print('percentage of common outstanding neuron diff2: {}'.format(len(common2) / len(diff2)))

    #common = np.intersect1d(both, ca_outstanding)
    #print('number of common outstanding neuron both: {}'.format(common))
    #print('percentage of common outstanding neuron both: {}'.format(len(common) / len(both)))
    
    end = time.time()
    print('Pre analysis time: {}'.format(end - start))

    return


def influence_estimation():
    if args.poison_type != 'semantic':
        print('Invalid poison type!')
        return

    _, _, test_clean_loader, test_adv_loader = \
        get_custom_loader(args.data_set, args.batch_size, args.poison_target, args.data_name, args.t_attack)

    if args.load_type == 'state_dict':
        net = getattr(models, args.arch)(num_classes=args.num_class, pretrained=0).to(device)

        state_dict = torch.load(args.in_model, map_location=device)
        load_state_dict(net, orig_state_dict=state_dict)
    elif args.load_type == 'model':
        net = torch.load(args.in_model, map_location=device)


    total_params = sum(p.numel() for p in net.parameters())
    print('Total number of parameters:{}'.format(total_params))

    start = time.time()
    # analyze hidden neurons
    if args.reanalyze:
        for each_class in range (0, args.num_class):
            print('Analyzing class:{}'.format(each_class))
            analyze_eachclass_influence(net, args.arch, each_class, args.num_class, args.num_sample, args.ana_layer, plot=args.plot)
    end = time.time()
    print('Influence estimation time: {}'.format(end - start))
    return


def hidden_plot():
    for each_class in range (0, args.num_class):
        print('Plotting class:{}'.format(each_class))
        hidden_test_all = []
        hidden_test_name = []
        for this_class in range(0, args.num_class):
            hidden_test_all_ = []
            for i in range(0, len(args.ana_layer)):
                hidden_test = np.loadtxt(
                    args.output_dir + "/test_pre0_" + "c" + str(this_class) + "_layer_" + str(args.ana_layer[i]) + ".txt")
                temp = hidden_test[:, [0, (this_class + 1)]]
                hidden_test_all_.append(temp)

            hidden_test_all.append(hidden_test_all_)

            hidden_test_name.append('class' + str(this_class))

        plot_multiple(hidden_test_all, hidden_test_name, each_class, args.ana_layer, save_n="test")


def analyze_eachclass(model, model_name, cur_class, num_class, num_sample, ana_layer, plot=False):
    '''
    use samples from base class, find important neurons
    '''
    clean_class_loader = get_custom_class_loader(args.data_set, args.batch_size, cur_class, args.data_name, args.t_attack)
    hidden_test = analyze_hidden(model, model_name, clean_class_loader, cur_class, num_sample, ana_layer)

    if plot:
        hidden_test_all = []
        hidden_test_name = []
        for this_class in range(0, num_class):
            hidden_test_all_ = []
            for i in range(0, len(ana_layer)):
                temp = hidden_test[i][:, [0, (this_class + 1)]]
                hidden_test_all_.append(temp)

            hidden_test_all.append(hidden_test_all_)

            hidden_test_name.append('class' + str(this_class))

        plot_multiple(hidden_test_all, hidden_test_name, cur_class, ana_layer, save_n="test")


def analyze_eachclass_influence(model, model_name, cur_class, num_class, num_sample, ana_layer, plot=False):
    '''
    use samples from base class, find important neurons
    '''
    clean_class_loader = get_custom_class_loader(args.data_set, args.batch_size, cur_class, args.data_name, args.t_attack)
    if args.inf_type == 'loo':  # leave one out
        analyze_hidden_influence(model, model_name, clean_class_loader, cur_class, num_sample, ana_layer)
    else:   # subsample
        analyze_hidden_influence_subsample(model, model_name, clean_class_loader, cur_class, num_sample, ana_layer,
                                           args.num_subgrp, args.cnt_per_grp)
def analyze_advclass(model, model_name, cur_class, num_class, num_sample, ana_layer, plot=False):
    '''
    use samples from base class, find important neurons
    '''
    adv_class_loader = get_data_adv_loader(args.data_set, args.batch_size, args.poison_target, args.data_name, args.t_attack)
    hidden_test = analyze_hidden(model, model_name, adv_class_loader, cur_class, num_sample, ana_layer)

    if plot:
        hidden_test_all = []
        hidden_test_name = []
        for this_class in range(0, num_class):
            hidden_test_all_ = []
            for i in range(0, len(ana_layer)):
                temp = hidden_test[i][:, [0, (this_class + 1)]]
                hidden_test_all_.append(temp)

            hidden_test_all.append(hidden_test_all_)

            hidden_test_name.append('class' + str(this_class))

        plot_multiple(hidden_test_all, hidden_test_name, cur_class, ana_layer, save_n="test")


def analyze_hidden(model, model_name, class_loader, cur_class, num_sample, ana_layer):
    out = []
    for cur_layer in ana_layer:
        model1, model2 = split_model(model, model_name, split_layer=cur_layer)
        model1.eval()
        model2.eval()

        do_predict_avg = []
        total_num_samples = 0
        for image, gt in class_loader:
            if total_num_samples >= num_sample:
                break

            image, gt = image.to(device), gt.to(device)

            # compute output
            with torch.no_grad():
                dense_output = model1(image)
                ori_output = model2(dense_output)
                #old_output = model(image)
                dense_hidden_ = torch.clone(torch.reshape(dense_output, (dense_output.shape[0], -1)))

                do_predict_neu = []
                #do convention for each neuron
                for i in range(0, len(dense_hidden_[0])):

                    hidden_do = dense_hidden_[:, i] + 1
                    dense_output_ = torch.clone(dense_hidden_)
                    dense_output_[:, i] = hidden_do

                    dense_output_ = torch.reshape(dense_output_, dense_output.shape)
                    dense_output_ = dense_output_.to(device)
                    output_do = model2(dense_output_).cpu().detach().numpy()
                    do_predict_neu.append(output_do)

                do_predict_neu = np.array(do_predict_neu)
                do_predict_neu = np.abs(ori_output.cpu().detach().numpy() - do_predict_neu)
                do_predict = np.mean(np.array(do_predict_neu), axis=1)

            do_predict_avg.append(do_predict)
            total_num_samples += len(gt)
        # average of all baches
        do_predict_avg = np.mean(np.array(do_predict_avg), axis=0)
        # insert neuron index
        idx = np.arange(0, len(do_predict_avg), 1, dtype=int)
        do_predict_avg = np.c_[idx, do_predict_avg]

        out.append(do_predict_avg)
        np.savetxt(args.output_dir + "/test_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt",
                   do_predict_avg, fmt="%s")

    return np.array(out)


def analyze_hidden_influence(model, model_name, class_loader, cur_class, num_sample, ana_layer):
    out = []

    for cur_layer in ana_layer:
        model1, model2 = split_model(model, model_name, split_layer=cur_layer)
        model1.eval()
        model2.eval()

        do_predict_avg = []
        total_num_samples = 0
        for image, gt in class_loader:
            if total_num_samples >= num_sample:
                break

            image, gt = image.to(device), gt.to(device)

            # compute output
            with torch.no_grad():
                dense_output = model1(image)
                ori_output = model2(dense_output)
                #old_output = model(image)
                dense_hidden_ = torch.clone(torch.reshape(dense_output, (dense_output.shape[0], -1)))

                do_predict_neu = []

                # each neuron LOO
                for i in range(0, len(dense_hidden_[0])):
                    hidden_do = dense_hidden_[:, i] * 0.0
                    dense_output_ = torch.clone(dense_hidden_)
                    dense_output_[:, i] = hidden_do

                    dense_output_ = torch.reshape(dense_output_, dense_output.shape)
                    dense_output_ = dense_output_.to(device)
                    output_do = model2(dense_output_).cpu().detach().numpy()
                    do_predict_neu.append(output_do)

                do_predict_neu = np.array(do_predict_neu)
                do_predict_neu = np.abs(ori_output.cpu().detach().numpy() - do_predict_neu)
                do_predict = np.mean(np.array(do_predict_neu), axis=1)

            do_predict_avg.append(do_predict)
            total_num_samples += len(gt)
        # average of all baches
        do_predict_avg = np.mean(np.array(do_predict_avg), axis=0)
        # insert neuron index
        idx = np.arange(0, len(do_predict_avg), 1, dtype=int)
        do_predict_avg = np.c_[idx, do_predict_avg]

        out.append(do_predict_avg)
        np.savetxt(args.output_dir + "/test_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt",
                   do_predict_avg, fmt="%s")

    return np.array(out)


def analyze_hidden_influence_subsample(model, model_name, class_loader, cur_class, num_sample, ana_layer,
                                       num_subgrp, cnt_per_grp):
    cur_layer = ana_layer[0]
    model1, model2 = split_model(model, model_name, split_layer=cur_layer)
    model1.eval()
    model2.eval()

    tot = get_neuron_count(model_name, cur_layer)
    all_cas = []
    total_num_samples = 0
    for image, gt in class_loader:
        if total_num_samples >= num_sample:
            break

        image, gt = image.to(device), gt.to(device)

        # compute output
        with torch.no_grad():
            dense_output = model1(image)
            dense_hidden_ = torch.clone(torch.reshape(dense_output, (dense_output.shape[0], -1)))

            cas = [] #cas of all neurons
            # each neuron subsample
            for neu_i in range(0, tot):

                pred_diff_avg = []  #pred_diff of all subgroups
                for i in range(0, num_subgrp):
                    idx = np.random.choice(np.setdiff1d(range(0, tot), neu_i), size=int(tot * cnt_per_grp),
                                           replace=False)
                    out_idx = list(idx)
                    in_idx = list(idx) + [neu_i]
                    ignore_idx_out = list(np.setdiff1d(range(0, tot), out_idx))
                    ignore_idx_in = list(np.setdiff1d(range(0, tot), in_idx))

                    dense_hidden_out = torch.clone(dense_hidden_)
                    dense_hidden_in = torch.clone(dense_hidden_)

                    dense_hidden_out[:, ignore_idx_out] = 0.0
                    dense_hidden_in[:, ignore_idx_in] = 0.0

                    dense_output_out = torch.reshape(dense_hidden_out, dense_output.shape)
                    dense_output_out = dense_output_out.to(device)
                    pred_out = model2(dense_output_out).cpu().detach().numpy()

                    dense_output_in = torch.reshape(dense_hidden_in, dense_output.shape)
                    dense_output_in = dense_output_in.to(device)
                    pred_in = model2(dense_output_in).cpu().detach().numpy()

                    pred_diff = np.abs(pred_out - pred_in)
                    pred_diff = np.mean(np.array(pred_diff), axis=0)
                    pred_diff_avg.append(pred_diff)
                cas.append(np.mean(np.array(pred_diff_avg), axis=0))

        total_num_samples += len(gt)
        all_cas.append(np.array(cas))

    all_cas = np.mean(np.array(all_cas), axis=0)
    # insert neuron index
    idx = np.arange(0, len(all_cas), 1, dtype=int)
    all_cas = np.c_[idx, all_cas]

    np.savetxt(args.output_dir + "/test_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt",
               all_cas, fmt="%s")

    return np.array(all_cas)


def analyze_hidden_influence_subsample_(model, model_name, class_loader, cur_class, num_sample, ana_layer,
                                       num_subgrp, cnt_per_grp):
    cur_layer = ana_layer[0]
    tot = get_neuron_count(model_name, cur_layer)
    all_cas = []
    for neu_i in range(0, tot):
        cas = []
        for i in range(0, num_subgrp):
            idx = np.random.choice(np.setdiff1d(range(0, tot), neu_i), size=int(tot * cnt_per_grp), replace=False)
            out_idx = list(idx)
            in_idx = list(idx) + [neu_i]
            ignore_idx_out = list(np.setdiff1d(range(0, tot), out_idx))
            ignore_idx_in = list(np.setdiff1d(range(0, tot), in_idx))
            model1, model2 = split_model(model, model_name, split_layer=cur_layer)
            model1.eval()
            model2.eval()

            pred_diff_avg = []
            total_num_samples = 0
            for image, gt in class_loader:
                if total_num_samples >= num_sample:
                    break

                image, gt = image.to(device), gt.to(device)

                # compute output
                with torch.no_grad():
                    dense_output = model1(image)
                    #old_output = model(image)
                    dense_hidden_ = torch.clone(torch.reshape(dense_output, (dense_output.shape[0], -1)))

                    dense_hidden_out = torch.clone(dense_hidden_)
                    dense_hidden_in = torch.clone(dense_hidden_)

                    dense_hidden_out[:, ignore_idx_out] = 0.0
                    dense_hidden_in[:, ignore_idx_in] = 0.0

                    dense_output_out = torch.reshape(dense_hidden_out, dense_output.shape)
                    dense_output_out = dense_output_out.to(device)
                    pred_out = model2(dense_output_out).cpu().detach().numpy()

                    dense_output_in = torch.reshape(dense_hidden_in, dense_output.shape)
                    dense_output_in = dense_output_in.to(device)
                    pred_in = model2(dense_output_in).cpu().detach().numpy()

                    pred_diff = np.abs(pred_out - pred_in)
                    pred_diff = np.mean(np.array(pred_diff), axis=0)

                pred_diff_avg.append(pred_diff)
                total_num_samples += len(gt)
            # average of all baches
            pred_diff_avg = np.mean(np.array(pred_diff_avg), axis=0)
            cas.append(pred_diff_avg)
        all_cas.append(np.mean(np.array(cas), axis=0))

    # insert neuron index
    idx = np.arange(0, len(all_cas), 1, dtype=int)
    all_cas = np.c_[idx, all_cas]

    np.savetxt(args.output_dir + "/test_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt",
               all_cas, fmt="%s")

    return np.array(all_cas)



def analyze_activation(model, model_name, class_loader, source, target, num_sample, ana_layer):
    for cur_layer in ana_layer:
        model1, model2 = split_model(model, model_name, split_layer=cur_layer)
        model1.eval()
        model2.eval()

        dense_output_avg = []
        total_num_samples = 0
        for image, gt in class_loader:
            if total_num_samples >= num_sample:
                break

            image, gt = image.to(device), gt.to(device)

            # compute output
            with torch.no_grad():
                dense_output = model1(image)
                dense_output = dense_output.cpu().detach().numpy()
                dense_output = np.mean(np.array(dense_output), axis=0)
            dense_output_avg.append(dense_output)
            total_num_samples += len(gt)
        # average of all baches
        dense_output_avg = np.mean(np.array(dense_output_avg), axis=0)  # 4096x10

        # insert neuron index
        idx = np.arange(0, len(dense_output_avg), 1, dtype=int)
        dense_output_avg = np.c_[idx, dense_output_avg]

        np.savetxt(args.output_dir + "/adv_act_" + "source_" + str(source) + "_target_" + str(target) + ".txt",
                   dense_output_avg, fmt="%s")

    return np.array(dense_output_avg)


def analyze_embedding(model, model_name, class_loader, source, target, num_sample, ana_layer):
    for cur_layer in ana_layer:
        model.eval()

        dense_output_avg = []
        total_num_samples = 0
        for image, gt in class_loader:
            if total_num_samples >= num_sample:
                break

            image, gt = image.to(device), gt.to(device)

            # compute output
            with torch.no_grad():
                dense_output = model(image)
                dense_output = dense_output.cpu().detach().numpy()
                dense_output = np.mean(np.array(dense_output), axis=0)
            dense_output_avg.append(dense_output)
            total_num_samples += len(gt)
        # average of all baches
        dense_output_avg = np.mean(np.array(dense_output_avg), axis=0)  # 4096x10

        # insert neuron index
        idx = np.arange(0, len(dense_output_avg), 1, dtype=int)
        dense_output_avg = np.c_[idx, dense_output_avg]

        np.savetxt(args.output_dir + "/adv_ce_" + "source_" + str(source) + "_target_" + str(target) + ".txt",
                   dense_output_avg, fmt="%s")

    return np.array(dense_output_avg)[target][1]


def analyze_pcc(num_class, ana_layer):
    pcc_class = []
    for source_class in range(0, num_class):
        print('analyzing pcc on class :{}'.format(source_class))
        for cur_layer in ana_layer:
            hidden_test_ = np.loadtxt(
                args.output_dir + "/test_pre0_" + "c" + str(source_class) + "_layer_" + str(cur_layer) + ".txt")
            hidden_test = np.insert(np.array(hidden_test_), 0, cur_layer, axis=1)
            hidden_test = np.array(hidden_test)

            pcc = []
            for i in range(0, num_class):
                if i == source_class:
                    continue

                # test check on average
                mat_ori = []
                for j in range(0, num_class):
                    if j == i:
                        continue
                    mat_j = hidden_test[:, (j + 2)]
                    mat_ori.append(mat_j)
                mat_ori = np.mean(np.array(mat_ori), axis=0)

                mat_cmp = hidden_test[:, (i + 2)]

                pcc_i = np.corrcoef(mat_ori, mat_cmp)[0, 1]
                pcc.append(pcc_i)
        pcc_class.append(pcc)
        np.savetxt(args.output_dir + "/pcc_" + "c" + str(source_class) + ".txt", pcc, fmt="%s")

    return pcc_class


def detect_pcc(num_class):
    pcc = []
    for source_class in range(0, num_class):
        pcc_class = np.loadtxt(args.output_dir + "/pcc_" + "c" + str(source_class) + ".txt")
        pcc_i = pcc_class
        pcc_i = np.insert(np.array(pcc_i), source_class, 0, axis=0)
        pcc.append(pcc_i)
    pcc_avg = np.mean(np.array(pcc), axis=0)
    pcc_avg = 1 - pcc_avg

    flag_list = outlier_detection(list(pcc_avg), max(pcc_avg), th=args.confidence)
    return flag_list


def analyze_source_class(net,  potential_target, num_class, ana_layer, num_sample):
    ce_cleans = []
    for source_class in range(0, num_class):
        for cur_layer in ana_layer:
            # clean class loader
            clean_class_loader = get_custom_class_loader(args.data_set, args.batch_size, source_class,
                                                         args.data_name,
                                                         args.t_attack)
            ce_clean = analyze_embedding(net, args.arch, clean_class_loader, source_class, potential_target, num_sample,
                                         args.ana_layer)

            if source_class == potential_target:
                ce_clean = .0

            ce_cleans.append(ce_clean)

    idx = np.argsort(ce_cleans)

    #print('[DEBUG]: ce_cleans{}'.format(idx))
    np.set_printoptions(precision=4)
    #print('[DEBUG]: ce_cleans{}'.format(np.array(ce_cleans)))

    # insert class index
    c_i = np.arange(0, len(idx), 1, dtype=int)
    temp = np.c_[c_i, ce_cleans]

    flag_list = outlier_detection(temp[:,1], max(temp[:,1]), th=args.confidence2, verbose=False)
    if len(flag_list) > 1:
        #print('[DEBUG] flag_list:{}'.format(flag_list))
        wanted = []
        for source_c, _ in flag_list:
            wanted.append(source_c)
        tgt_to_src = np.loadtxt(args.output_dir + "/adv_ce_" + "source_" + str(potential_target) + "_target_" + str(potential_target) + ".txt")
        temp = tgt_to_src[wanted]
        ind = np.argsort(temp[:, 1])[::-1]
        temp = temp[ind]
        out = int(temp[-1][0])
    else:
        out = idx[-1]
    return out


def locate_outstanding_neuron(potential_source, potential_target, cur_layer):
    # load sensitive neuron
    hidden_test = np.loadtxt(
        args.output_dir + "/test_pre0_" + "c" + str(potential_source) + "_layer_" + str(cur_layer) + ".txt")
    # check common important neuron
    temp = hidden_test[:, [0, (potential_target + 1)]]
    ind = np.argsort(temp[:, 1])[::-1]
    temp = temp[ind]
    np.savetxt(args.output_dir + "/outstanding_" + "c" + str(potential_source) + "_target_" +
               str(potential_target) + ".txt", temp[:, 0].astype(int), fmt="%s")
    return temp[:, 0]

def solve_detect_common_outstanding_neuron(num_class, ana_layer):
        '''
        find common outstanding neurons
        return potential attack base class and target class
        '''

        top_list = []
        top_neuron = []

        for each_class in range (0, num_class):
            top_list_i, top_neuron_i = detect_eachclass_all_layer(each_class, num_class, ana_layer)
            top_list = top_list + top_list_i
            top_neuron.append(top_neuron_i)

        flag_list = outlier_detection(top_list, max(top_list))
        if len(flag_list) == 0:
            return []

        base_class, target_class = find_target_class(flag_list, num_class)

        ret = []
        for i in range(0, len(base_class)):
            ret.append([base_class[i], target_class[i]])

        # remove classes that are natualy alike
        remove_i = []
        for i in range(0, len(base_class)):
            if base_class[i] in target_class:
                ii = target_class.index(base_class[i])
                if target_class[i] == base_class[ii]:
                    remove_i.append(i)

        out = [e for e in ret if ret.index(e) not in remove_i]
        if len(out) > 3:
            out = out[:3]
        return out


def detect_eachclass_all_layer(cur_class, num_class, ana_layer):
        hidden_test = []
        for cur_layer in ana_layer:
            hidden_test_ = np.loadtxt(args.output_dir + "/test_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt")
            hidden_test_ = np.insert(np.array(hidden_test_), 0, cur_layer, axis=1)
            hidden_test = hidden_test + list(hidden_test_)

        hidden_test = np.array(hidden_test)
        # check common important neuron
        temp = hidden_test[:, [0, 1, (cur_class + 2)]]
        ind = np.argsort(temp[:,2])[::-1]
        temp = temp[ind]

        # find outlier hidden neurons
        top_num = len(outlier_detection(temp[:, 2], max(temp[:, 2]), verbose=False))
        num_neuron = top_num
        print('significant neuron: {}'.format(num_neuron))
        cur_top = list(temp[0: (num_neuron - 1)][:, [0, 1]])

        top_list = []
        top_neuron = []
        # compare with all other classes
        for cmp_class in range(0, num_class):
            if cmp_class == cur_class:
                top_list.append(0)
                top_neuron.append(np.array([0] * num_neuron))
                continue
            temp = hidden_test[:, [0, 1, (cmp_class + 2)]]
            ind = np.argsort(temp[:, 2])[::-1]
            temp = temp[ind]
            cmp_top = list(temp[0: num_neuron][:, [0, 1]])
            temp = np.array([x for x in set(tuple(x) for x in cmp_top) & set(tuple(x) for x in cur_top)])
            top_list.append(len(temp))
            top_neuron.append(temp)

        # top_list: number of intersected neurons (10,)
        # top_neuron: layer and index of intersected neurons    ((2, n) x 10)
        return list(np.array(top_list) / top_num), top_neuron


def solve_detect_outlier(num_class, ana_layer):
    '''
    analyze outliers to certain class, find potential backdoor due to overfitting
    '''
    #print('Detecting outliers.')

    tops = []   #outstanding neuron for each class

    for each_class in range (0, num_class):
        top_ = find_outstanding_neuron(each_class, num_class, ana_layer, prefix="")
        tops.append(top_)

    save_top = []
    for top in tops:
        save_top = [*save_top, *top]
    save_top = np.array(save_top)
    flag_list = outlier_detection(1 - save_top/max(save_top), 1)
    np.savetxt(args.output_dir + "/outlier_count.txt", save_top, fmt="%s")

    base_class, target_class = find_target_class(flag_list, num_class)

    out = []
    for i in range (0, len(base_class)):
        if base_class[i] != target_class[i]:
            out.append([base_class[i], target_class[i]])

    ret = []
    base_class = []
    target_class = []
    for i in range(0, len(out)):
        base_class.append(out[i][0])
        target_class.append(out[i][1])
        ret.append([base_class[i], target_class[i]])

    remove_i = []
    for i in range(0, len(base_class)):
        if base_class[i] in target_class:
            ii = target_class.index(base_class[i])
            if target_class[i] == base_class[ii]:
                remove_i.append(i)

    out = [e for e in ret if ret.index(e) not in remove_i]
    if len(out) > 1:
        out = out[:1]
    return out


def find_outstanding_neuron(cur_class, num_class, ana_layer, prefix=""):
    '''
    find outstanding neurons for cur_class
    '''
    hidden_test = []
    for cur_layer in ana_layer:
        hidden_test_ = np.loadtxt(args.output_dir + '/' + prefix + "test_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt")
        hidden_test_ = np.insert(np.array(hidden_test_), 0, cur_layer, axis=1)
        hidden_test = hidden_test + list(hidden_test_)
    hidden_test = np.array(hidden_test)

    # find outlier hidden neurons for all class embedding
    top_num = []
    # compare with all other classes
    for cmp_class in range (0, num_class):
        temp = hidden_test[:, [0, 1, (cmp_class + 2)]]
        ind = np.argsort(temp[:,1])[::-1]
        temp = temp[ind]
        cmp_top = outlier_detection_overfit(temp[:, (2)], max(temp[:, (2)]), verbose=False)
        top_num.append((cmp_top))

    return top_num


def outlier_detection(cmp_list, max_val, th=2, verbose=False):
        cmp_list = list(np.array(cmp_list) / max_val)
        consistency_constant = 1.4826  # if normal distribution
        median = np.median(cmp_list)
        mad = consistency_constant * np.median(np.abs(cmp_list - median))   #median of the deviation
        min_mad = np.abs(np.min(cmp_list) - median) / mad

        flag_list = []
        i = 0
        for cmp in cmp_list:
            if cmp_list[i] < median:
                i = i + 1
                continue
            if np.abs(cmp_list[i] - median) / mad > th:
                flag_list.append((i, cmp_list[i]))
            i = i + 1

        if len(flag_list) > 0:
            flag_list = sorted(flag_list, key=lambda x: x[1])
            if verbose:
                print('flagged label list: %s' %
                      ', '.join(['%d: %2f' % (idx, val)
                                 for idx, val in flag_list]))
        return flag_list
        pass


def outlier_detection_overfit(cmp_list, max_val, verbose=True):
    flag_list = outlier_detection(cmp_list, max_val, verbose)
    return len(flag_list)


def find_target_class(flag_list, num_class):
        if len(flag_list) == 0:
            return [[],[]]

        a_flag = np.array(flag_list)

        ind = np.argsort(a_flag[:,1])[::-1]
        a_flag = a_flag[ind]

        base_classes = []
        target_classes = []

        i = 0
        for (flagged, mad) in a_flag:
            base_class = int(flagged / num_class)
            target_class = int(flagged - num_class * base_class)
            base_classes.append(base_class)
            target_classes.append(target_class)
            i = i + 1

        return base_classes, target_classes


def plot_multiple(_rank, name, cur_class, ana_layer, normalise=False, save_n=""):
    # plot the permutation of cmv img and test imgs
    plt_row = len(_rank)

    rank = []
    for _rank_i in _rank:
        rank.append(copy.deepcopy(_rank_i))

    plt_col = len(ana_layer)
    fig, ax = plt.subplots(plt_row, plt_col, figsize=(7 * plt_col, 5 * plt_row), sharex=False, sharey=True)

    if plt_col == 1:
        col = 0
        for do_layer in ana_layer:
            for row in range(0, plt_row):
                # plot ACE
                if row == 0:
                    ax[row].set_title('Layer_' + str(do_layer))
                    # ax[row, col].set_xlabel('neuron index')
                    # ax[row, col].set_ylabel('delta y')

                if row == (plt_row - 1):
                    # ax[row, col].set_title('Layer_' + str(do_layer))
                    ax[row].set_xlabel('neuron index')

                ax[row].set_ylabel(name[row])

                # Baseline is np.mean(expectation_do_x)
                if normalise:
                    rank[row][col][:, 1] = rank[row][col][:, 1] / np.max(rank[row][col][:, 1])

                ax[row].scatter(rank[row][col][:, 0].astype(int), rank[row][col][:, 1],
                                     label=str(do_layer) + '_cmv',
                                     color='b')
                ax[row].legend()

            col = col + 1
    else:
        col = 0
        for do_layer in ana_layer:
            for row in range(0, plt_row):
                # plot ACE
                if row == 0:
                    ax[row, col].set_title('Layer_' + str(do_layer))
                    # ax[row, col].set_xlabel('neuron index')
                    # ax[row, col].set_ylabel('delta y')

                if row == (plt_row - 1):
                    # ax[row, col].set_title('Layer_' + str(do_layer))
                    ax[row, col].set_xlabel('neuron index')

                ax[row, col].set_ylabel(name[row])

                # Baseline is np.mean(expectation_do_x)
                if normalise:
                    rank[row][col][:, 1] = rank[row][col][:, 1] / np.max(rank[row][col][:, 1])

                ax[row, col].scatter(rank[row][col][:, 0].astype(int), rank[row][col][:, 1], label=str(do_layer) + '_cmv',
                                     color='b')
                ax[row, col].legend()

            col = col + 1
    if normalise:
        plt.savefig(args.output_dir + "/plt_n_c" + str(cur_class) + save_n + ".png")
    else:
        plt.savefig(args.output_dir + "/plt_c" + str(cur_class) + save_n + ".png")
    # plt.show()


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    #'''
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255

    x = np.clip(x, 0, 255).astype('uint8')
    '''
    x = np.clip(x, 0, 1)
    '''
    return x


def train(model, criterion, optimizer, data_loader):
    model.train()
    total_correct = 0
    total_loss = 0.0
    for i, (images, labels) in enumerate(data_loader):
        labels = labels.long()
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)

        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


def train_tune(model, criterion, reg, target_class, optimizer, data_loader, adv_loader):
    model.train()
    total_correct = 0
    total_loss = 0.0

    needed = 20
    adv_iter = iter(adv_loader)
    for i, (images, labels) in enumerate(data_loader):
        # select adv samples
        images_adv = None
        labels_adv = None
        while images_adv is None or len(images_adv) < needed:
            images_adv_, labels_adv_ = next(adv_iter, (None, None))
            if images_adv_ is None:
                adv_iter = iter(adv_loader)
                images_adv_, labels_adv_ = next(adv_iter, (None, None))
            if images_adv is None:
                images_adv = images_adv_
                labels_adv = labels_adv_
            else:
                images_adv = torch.cat((images_adv, images_adv_), 0)
                labels_adv = torch.cat((labels_adv, labels_adv_), 0)
        images_adv = images_adv[:needed]
        labels_adv = labels_adv[:needed]

        _input = torch.cat((images[:(args.batch_size - needed)],
                            images_adv), 0)
        _output = torch.cat((labels[:(args.batch_size - needed)],
                             labels_adv), 0)
        images = _input
        labels = _output

        labels = labels.long()
        images, labels = images.to(device), labels.to(device)
        target = (torch.ones(needed, dtype=torch.int64) * target_class).to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels) - reg * criterion(output[-needed:], target)

        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()

        loss.backward()
        optimizer.step()


    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


def test(model, criterion, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            labels = labels.long()
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


if __name__ == '__main__':
    if args.option == 'causality_analysis':
        causality_analysis()
    elif args.option == 'test':
        run_test()
    elif args.option == 'plot':
        hidden_plot()
    elif args.option == 'detect':
        detect()
    elif args.option == 'remove':
        remove()
    elif args.option == 'gen_trigger':
        gen_trigger()
    elif args.option == 'pre_analysis':
        pre_analysis(0)
    elif args.option == 'pre_ana_ifl':
        pre_analysis(1)
    elif args.option == 'influence':
        influence_estimation()
    elif args.option == 'dfr':
        dfr()
