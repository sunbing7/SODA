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

#from numpy import dot
#from numpy.linalg import norm

from models.split_model import split_model, reconstruct_model, recover_model, get_neuron_count

torch.manual_seed(123)
parser = argparse.ArgumentParser(description='Semantic backdoor mitigation.')

# Basic model parameters.
parser.add_argument('--arch', type=str, default='resnet18',
                    choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'MobileNetV2', 'vgg19_bn', 'vgg11_bn'])
parser.add_argument('--batch_size', type=int, default=128, help='the batch size for dataloader')
parser.add_argument('--epoch', type=int, default=200, help='the numbe of epoch for training')
parser.add_argument('--save_every', type=int, default=20, help='save checkpoints every few epochs')
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
                                                                     'gen_trigger', 'test', 'pre_analysis'], help='run option')
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

    _, _, _, test_clean_loader, test_adv_loader = \
        get_custom_loader(args.data_set, args.batch_size, args.poison_target, args.data_name, args.t_attack)
    if args.test_reverse:
        radv_loader = get_data_adv_loader(args.data_dir, is_train=False, batch_size=args.batch_size,
                                          t_target=args.poison_target, dataset=args.data_name, t_attack=args.t_attack, option='reverse')

    poison_test_loader = test_adv_loader
    clean_test_loader = test_clean_loader

    if args.load_type == 'state_dict':
        net = getattr(models, args.arch)(num_classes=args.num_class).to(device)

        state_dict = torch.load(args.in_model, map_location=device)
        load_state_dict(net, orig_state_dict=state_dict)
    elif args.load_type == 'model':
        net = torch.load(args.in_model, map_location=device)

    #summary(net, (3, 32, 32))
    #print(net)

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

    _, _, _, test_clean_loader, test_adv_loader = \
        get_custom_loader(args.data_set, args.batch_size, args.poison_target, args.data_name, args.t_attack)

    poison_test_loader = test_adv_loader
    clean_test_loader = test_clean_loader

    if args.load_type == 'state_dict':
        net = getattr(models, args.arch)(num_classes=args.num_class).to(device)

        state_dict = torch.load(args.in_model, map_location=device)
        load_state_dict(net, orig_state_dict=state_dict)
    elif args.load_type == 'model':
        net = torch.load(args.in_model, map_location=device)
    #summary(net, (3, 32, 32))
    #print(net)

    total_params = sum(p.numel() for p in net.parameters())
    print('Total number of parameters:{}'.format(total_params))

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    logger.info('Epoch \t lr \t Time \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    #torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_init.th'))
    cl_loss, cl_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
    po_loss, po_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
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
    #print('pcc flag list: {}'.format(flag_list))
    if len(flag_list) == 0:# or (args.potential_target != 'na' and int(args.potential_target) != flag_list[-1][0]):
        print('No semantic backdoor detected!')
        print('Detection time:{}'.format(end1 - start))
        return
    potential_target = flag_list[-1][0]

    # Step 2 find source class
    if args.load_type == 'state_dict':
        net = getattr(models, args.arch)(num_classes=args.num_class).to(device)

        state_dict = torch.load(args.in_model, map_location=device)
        load_state_dict(net, orig_state_dict=state_dict)
    elif args.load_type == 'model':
        net = torch.load(args.in_model, map_location=device)

    #summary(net, (3, 32, 32))
    #print(net)

    flag_list = analyze_source_class2(net, args.arch, args.poison_target, potential_target, args.num_class, args.ana_layer, args.num_sample, args.confidence2)
    end2 = time.time()
    print('[Detection] potential source class: {}, target class: {}'.format(int(flag_list), int(potential_target)))
    print('Detection time:{}'.format(end2 - start))
    return


def remove():
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

    _, train_clean_loader, _, test_clean_loader, test_adv_loader = \
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
        net = getattr(models, args.arch)(num_classes=args.num_class).to(device)

        state_dict = torch.load(args.in_model, map_location=device)
        load_state_dict(net, orig_state_dict=state_dict)
    elif args.load_type == 'model':
        net = torch.load(args.in_model, map_location=device)
    mask = np.zeros(get_neuron_count(args.arch))
    neu_idx = np.loadtxt(args.output_dir + "/outstanding_" + "c" + str(args.potential_source) + "_target_" + str(args.poison_target) + ".txt")
    neu_idx = neu_idx[:int(len(neu_idx) * args.top)]
    mask[neu_idx.astype(int)] = 1
    mask = torch.from_numpy(mask).to(device)
    net = reconstruct_model(net, args.arch, mask, split_layer=args.ana_layer[0])

    #summary(net, (3, 32, 32))
    #print(net)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    #'''
    logger.info('Epoch \t lr \t Time \t PoisonLoss \t PoisonACC \t RPoisonLoss \t RPoisonACC \t CleanLoss \t CleanACC')
    torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_init.th'))
    cl_loss, cl_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
    po_loss, po_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
    rpo_loss, rpo_acc = test(model=net, criterion=criterion, data_loader=radv_loader_test)
    logger.info('0 \t None \t None \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(po_loss, po_acc, rpo_loss, rpo_acc, cl_loss, cl_acc))
    #'''
    for epoch in range(1, args.epoch):
        start = time.time()
        _adjust_learning_rate(optimizer, epoch, args.lr)
        lr = optimizer.param_groups[0]['lr']

        train_loss, train_acc = train_tune(model=net, criterion=criterion, reg=args.reg, target_class=args.poison_target, optimizer=optimizer,
                                           data_loader=train_clean_loader, adv_loader=radv_loader)

        cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
        po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
        rpo_loss, rpo_acc = test(model=net, criterion=criterion, data_loader=radv_loader_test)

        end = time.time()
        logger.info(
            '%d \t %.3f \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
            epoch, lr, end - start, po_test_loss, po_test_acc, rpo_loss, rpo_acc,
            cl_test_loss, cl_test_acc)


        if (epoch + 1) % args.save_every == 0:
            torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_finetune4_{}_{}.th'.format(args.t_attack, epoch)))

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
    torch.save(rnet, os.path.join(args.output_dir, 'model_finetune4_' + str(args.t_attack) + '_last.th'))
    #'''

    return


def gen_trigger():
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

    clean_class_loader = get_custom_class_loader(args.data_set, args.batch_size, args.potential_source, args.data_name,
                                                 args.t_attack, is_train=True)

    if args.load_type == 'state_dict':
        net = getattr(models, args.arch)(num_classes=args.num_class).to(device)

        state_dict = torch.load(args.in_model, map_location=device)
        load_state_dict(net, orig_state_dict=state_dict)

    elif args.load_type == 'model':
        net = torch.load(args.in_model, map_location=device)

    net.requires_grad = False
    net.eval()

    #for all samples
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

            criterion = torch.nn.CrossEntropyLoss().to(device)
            optimizer = torch.optim.SGD([image], lr=args.lr, momentum=0.9, weight_decay=5e-4)

            for epoch in range(0, int(args.epoch / 1)):
                start = time.time()
                #image_batch = image.repeat(args.batch_size, 1, 1, 1)
                #out = net(image_batch)
                #target = (torch.ones(image_batch.shape[0], dtype=torch.int64) * args.poison_target).to(device)
                out = net(image.reshape(torch.unsqueeze(image, 0).shape))
                target = (torch.Tensor([args.poison_target]).long()).to(device)
                #loss = criterion(out, target)
                loss = - torch.mean(out[:, args.poison_target]) + args.reg * torch.mean(torch.square(image))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                target_prediction = torch.softmax(out, dim=1)[0, args.poison_target]
                #'''
                if (epoch + 1) % 500 == 0:
                    source_prediction = torch.softmax(out, dim=1)[0, args.potential_source]
                    print("Iteration %d, Loss=%f, target prob=%f, source prob=%f" % (
                        epoch, float(loss), float(target_prediction), float(source_prediction)))
                #'''
                if target_prediction >= 0.9:
                    break
            predict = net(image.reshape(torch.unsqueeze(image, 0).shape))
            #print('prediction: {}'.format(predict))
            predict = torch.argmax(predict)
            print('prediction: {}'.format(predict))

            #image = image_batch[0]#torch.mean(image_batch, 0)
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


def pre_analysis():
    '''
    look at outstanding neuron of adv sample and CA
    '''
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
        net = getattr(models, args.arch)(num_classes=args.num_class).to(device)

        state_dict = torch.load(args.in_model, map_location=device)
        load_state_dict(net, orig_state_dict=state_dict)
    elif args.load_type == 'model':
        net = torch.load(args.in_model, map_location=device)
    #summary(net, (3, 32, 32))
    #print(net)

    total_params = sum(p.numel() for p in net.parameters())
    print('Total number of parameters:{}'.format(total_params))

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    start = time.time()
    # analyze hidden neuron activation on infected samples
    if args.reanalyze:
        #data_file, is_train=False, batch_size=64, t_target=6, dataset='CIFAR10', t_attack='green', option='original'
        adv_class_loader = get_data_adv_loader(args.data_set, batch_size=args.batch_size, t_target=args.poison_target, dataset=args.data_name,
                                               t_attack=args.t_attack, option='original')
        act = analyze_activation(net, args.arch, adv_class_loader, args.potential_source, args.potential_target, args.num_sample, args.ana_layer)

        #clean class loader
        clean_class_loader = get_custom_class_loader(args.data_set, args.batch_size, args.potential_source, args.data_name,
                                                     args.t_attack)
        act_clean = analyze_activation(net, args.arch, clean_class_loader, args.potential_source, args.potential_target,
                                 args.num_sample, args.ana_layer)

    act_outstanding = np.array(outlier_detection(act[:, 1], max(act[:, 1]), th=args.confidence, verbose=False))[:,0]
    print('activation adv outstanding count: {}'.format(len(act_outstanding)))
    #print('act_outstanding:{}'.format(act_outstanding))

    act_clean_outstanding = np.array(outlier_detection(act_clean[:, 1], max(act_clean[:, 1]), th=args.confidence, verbose=False))[:,0]
    #print('act_clean_outstanding:{}'.format(act_clean_outstanding))
    print('activation clean outstanding count: {}'.format(len(act_clean_outstanding)))

    # union
    both = np.unique(np.concatenate((act_outstanding, act_clean_outstanding), 0))
    #print('adv and act: {}'.format(both))

    # yields the elements in `act_outstanding` that are NOT in `act_clean_outstanding`
    diff = np.setdiff1d(act_outstanding, act_clean_outstanding)
    print('different outstanding neuron between adv and act: {}'.format(diff))
    print('number of different outstanding neuron between adv and act: {}'.format(len(diff)))
    #common = np.intersect1d(act_outstanding, act_clean_outstanding)#np.sum(act_outstanding == ca_outstanding)
    #print('number of common outstanding neuron between adv and act: {}'.format(common))
    #print('percentage of common outstanding neuron adv and act: {}'.format(len(common) / len(act_outstanding)))
    #print('clean outstanding count: {}'.format(len(act_clean_outstanding)))

    #mat_cmp = act[:, 1]
    #mat_ori = act_clean[:, 1]
    #pcc_i = np.corrcoef(mat_ori, mat_cmp)[0, 1]
    #print('pcc adv and clean: {}'.format(pcc_i))

    # analyze hidden neuron causal attribution
    clean_class_loader = get_custom_class_loader(args.data_set, args.batch_size, args.potential_source, args.data_name, args.t_attack)
    analyze_hidden(net, args.arch, clean_class_loader, args.potential_source, args.num_sample, args.ana_layer)

    hidden_test = np.loadtxt(
        args.output_dir + "/test_pre0_" + "c" + str(args.potential_source) + "_layer_" + str(args.ana_layer[0]) + ".txt")

    temp = hidden_test[:, [0, (int(args.potential_target) + 1)]]
    np.savetxt(args.output_dir + "/adv_ca_" + "source_" + str(args.potential_source) + "_target_" + str(args.potential_target) + ".txt",
               temp, fmt="%s")
    ca_outstanding = np.array(outlier_detection(temp[:, 1], max(temp[:, 1]), th=args.confidence2, verbose=False))[:,0]
    #print('ca_outstanding:{}'.format(ca_outstanding))
    #common = np.intersect1d(act_outstanding, ca_outstanding)#np.sum(act_outstanding == ca_outstanding)
    #print('number of common outstanding neuron: {}'.format(common))
    #print('percentage of common outstanding neuron: {}'.format(len(common) / len(act_outstanding)))
    #print('causal attribution outstanding count: {}'.format(len(ca_outstanding)))

    common = np.intersect1d(diff, ca_outstanding)
    print('common outstanding neuron diff: {}'.format(common))
    print('number of common outstanding neuron diff: {}'.format(len(common)))
    print('percentage of common outstanding neuron diff: {}'.format(len(common) / len(diff)))

    #common = np.intersect1d(both, ca_outstanding)
    #print('number of common outstanding neuron both: {}'.format(common))
    #print('percentage of common outstanding neuron both: {}'.format(len(common) / len(both)))
    
    #pcc analysis
    mask1 = np.zeros(len(temp))
    mask1[list(act_outstanding.astype(int))] = 1

    mask2 = np.zeros(len(temp))
    mask2[list(ca_outstanding.astype(int))] = 1

    #mat_cmp = act[:, 1] * mask1
    #mat_ori = hidden_test[:, (int(args.potential_target) + 1)] * mask2
    #pcc_i = np.corrcoef(mat_ori, mat_cmp)[0, 1]
    #print('pcc: {}'.format(pcc_i))
    end = time.time()
    print('Pre analysis time: {}'.format(end - start))

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
        #print('current layer: {}'.format(cur_layer))
        model1, model2 = split_model(model, model_name, split_layer=cur_layer)
        model1.eval()
        model2.eval()
        #summary(model1, (3, 32, 32))
        #summary(model2, (128, 16, 16))
        do_predict_avg = []
        total_num_samples = 0
        for image, gt in class_loader:
            if total_num_samples >= num_sample:
                break

            image, gt = image.to(device), gt.to(device)

            # compute output
            with torch.no_grad():
                dense_output = model1(image)
                #dense_output = dense_output.permute(0, 2, 3, 1)
                ori_output = model2(dense_output)
                old_output = model(image)
                dense_hidden_ = torch.clone(torch.reshape(dense_output, (dense_output.shape[0], -1)))
                #ori_output = filter_model(image)
                do_predict_neu = []
                do_predict = []
                #do convention for each neuron
                for i in range(0, len(dense_hidden_[0])):
                    # x2
                    #hidden_do = np.ones(shape=dense_hidden_[:, i].shape)
                    #hidden_do = torch.from_numpy(hidden_do).to(device)
                    hidden_do = dense_hidden_[:, i] + 1
                    dense_output_ = torch.clone(dense_hidden_)
                    dense_output_[:, i] = hidden_do

                    dense_output_ = torch.reshape(dense_output_, dense_output.shape)
                    dense_output_ = dense_output_.to(device)
                    output_do = model2(dense_output_).cpu().detach().numpy()
                    do_predict_neu.append(output_do) # 4096x32x10
                    '''
                    hidden_do = np.zeros(shape=dense_hidden_[:, i].shape)
                    dense_output_ = torch.clone(dense_hidden_)
                    dense_output_[:, i] = torch.from_numpy(hidden_do)
                    dense_output_ = torch.reshape(dense_output_, dense_output.shape)
                    dense_output_ = dense_output_.to(device)
                    output_do = model2(dense_output_).cpu().detach().numpy()
                    do_predict_neu.append(output_do) # 4096x32x10
                    '''
                do_predict_neu = np.array(do_predict_neu)
                do_predict_neu = np.abs(ori_output.cpu().detach().numpy() - do_predict_neu)
                do_predict = np.mean(np.array(do_predict_neu), axis=1)  #4096x10

            do_predict_avg.append(do_predict) #batchx4096x11
            total_num_samples += len(gt)
        # average of all baches
        do_predict_avg = np.mean(np.array(do_predict_avg), axis=0) #4096x10
        # insert neuron index
        idx = np.arange(0, len(do_predict_avg), 1, dtype=int)
        do_predict_avg = np.c_[idx, do_predict_avg]
        #out = do_predict_avg[:, [0, (target_class + 1)]]
        out.append(do_predict_avg)
        np.savetxt(args.output_dir + "/test_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt",
                   do_predict_avg, fmt="%s")

    return np.array(out)


def analyze_activation(model, model_name, class_loader, source, target, num_sample, ana_layer):
    for cur_layer in ana_layer:
        #print('current layer: {}'.format(cur_layer))
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
                #dense_hidden_ = torch.clone(torch.reshape(dense_output, (dense_output.shape[0], -1)))
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


def analyze_pcc(num_class, ana_layer):
    pcc_class = []
    for source_class in range(0, num_class):
        print('analyzing pcc on class :{}'.format(source_class))
        hidden_test = []
        for cur_layer in ana_layer:
            hidden_test_ = np.loadtxt(
                args.output_dir + "/test_pre0_" + "c" + str(source_class) + "_layer_" + str(cur_layer) + ".txt")
            # l = np.ones(len(hidden_test_)) * cur_layer
            hidden_test = np.insert(np.array(hidden_test_), 0, cur_layer, axis=1)
            #hidden_test = hidden_test + list(hidden_test_)

            hidden_test = np.array(hidden_test)

            pcc = []

            #mat_ori = hidden_test[:, (source_class + 2)]
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
                #cos similarity
                #pcc_i = dot(mat_ori, mat_cmp) / (norm(mat_ori) * norm(mat_cmp))
                pcc.append(pcc_i)
        pcc_class.append(pcc)
        np.savetxt(args.output_dir + "/pcc_" + "c" + str(source_class) + ".txt", pcc, fmt="%s")

    return pcc_class


def detect_pcc(num_class):
    pcc = []
    for source_class in range(0, num_class):
        pcc_class = np.loadtxt(args.output_dir + "/pcc_" + "c" + str(source_class) + ".txt")
        #pcc_i = pcc_class[-1, :]
        pcc_i = pcc_class
        pcc_i = np.insert(np.array(pcc_i), source_class, 0, axis=0)
        pcc.append(pcc_i)
    pcc_avg = np.mean(np.array(pcc), axis=0)
    pcc_avg = 1 - pcc_avg
    #find outlier

    flag_list = outlier_detection(list(pcc_avg), max(pcc_avg), th=args.confidence)
    return flag_list


def analyze_source_class(model, model_name, target_class, potential_target, num_class, ana_layer, num_sample, th=3):
    out = []
    old_out = []
    for source_class in range(0, num_class):
        print('analyzing source class: {}'.format(source_class))
        class_loader = get_custom_class_loader(args.data_set, args.batch_size, source_class, args.data_name, target_class)
        for cur_layer in ana_layer:
            # load sensitive neuron
            hidden_test = np.loadtxt(
                args.output_dir + "/test_pre0_" + "c" + str(source_class) + "_layer_" + str(cur_layer) + ".txt")
            # check common important neuron
            temp = hidden_test[:, [0, (potential_target + 1)]]
            ind = np.argsort(temp[:, 1])[::-1]
            temp = temp[ind]

            # find outlier hidden neurons
            top_num = int(len(outlier_detection(temp[:, 1], max(temp[:, 1]), th=th, verbose=False)))
            top_neuron = list(temp[:top_num].T[0].astype(int))
            np.savetxt(args.output_dir + "/outstanding_" + "c" + str(source_class) + "_target_" + str(potential_target) + ".txt",
                       temp[:,0].astype(int), fmt="%s")
            #print('significant neuron: {}'.format(top_num))
            '''
            # get source to source top neuron
            temp_s = hidden_test[:, [0, (source_class + 1)]]
            ind = np.argsort(temp_s[:, 1])[::-1]
            temp_s = temp_s[ind]

            # find outlier hidden neurons
            top_num_s = int(len(outlier_detection(temp_s[:, 1], max(temp_s[:, 1]), verbose=False)))
            top_neuron_s = list(temp_s[:top_num_s].T[0].astype(int))

            ca = Counter(top_neuron)
            cb= Counter(top_neuron_s)
            diff = sorted((ca - cb).elements())
            print('significant neuron: {}, fraction: {}'.format(len(diff), len(diff)/top_num))
            '''
            #top_neuron = diff
            #np.savetxt(args.output_dir + "/sensitive" + "c" + str(source_class) + "_target_" + str(potential_target) + ".txt",
            #           top_neuron, fmt="%s")
            #top_neuron = [24,429,297,401,96,459,246,367,91,509,445,287,320,291,182,198,474,47,308,113,253,290,276,476,73,220,505,105,144,410,319,141,212,15,81,5,275,448,185,89,337,173,1,214,493,176,12,265,458,87,322,331,56,384,400,54,145,243,97,51,109,510,465,369,83,330,126,497,292,157,324,247,484,499,306,372,390,427,127,295,16,354,230,72,86,371,332,422,502,67,500,356,115,314,99,231,450,368,187,441,211,340,169,472,263,155,160,238,192,71,226]
            #prepare mask
            mask = np.zeros(len(temp))
            mask[top_neuron] = 1
            mask = torch.from_numpy(mask).to(device)

            model1, model2 = split_model(model, model_name, split_layer=cur_layer)
            model1.eval()
            model2.eval()

            do_predict_avg = []
            old_predict_avg = []
            total_num_samples = 0
            for image, gt in class_loader:
                if total_num_samples >= num_sample:
                    break

                image, gt = image.to(device), gt.to(device)

                # compute output
                with torch.no_grad():
                    dense_output = model1(image)
                    # dense_output = dense_output.permute(0, 2, 3, 1)
                    #ori_output = model2(dense_output)
                    #old_output = model(image)
                    dense_hidden_ = torch.clone(torch.reshape(dense_output, (dense_output.shape[0], -1)))
                    # ori_output = filter_model(image)
                    do_predict_neu = []
                    do_predict = []
                    # do convention for each neuron

                    hidden_do = (mask * dense_hidden_).cpu().detach().numpy()
                    #dense_output_ = torch.clone(dense_hidden_)
                    #dense_output_ = dense_output_ + hidden_do
                    #dense_output_ = torch.reshape(dense_output_, dense_output.shape)
                    #dense_output_ = dense_output_.to(device)
                    #output_do = model2(dense_output_.float()).cpu().detach().numpy()
                    do_predict = np.mean(np.array(hidden_do), axis=0)
                    #old_predict = np.mean(old_output.cpu().detach().numpy(), axis=0)

                do_predict_avg.append(do_predict)  # batchx4096x11
                #old_predict_avg.append(old_predict)
                total_num_samples += len(gt)
            # average of all baches
            do_predict_avg = np.mean(np.array(do_predict_avg), axis=0)  # 4096x10
            #do_predict_avg = np.insert(np.array(do_predict_avg), 0, source_class, axis=0)
            #old_predict_avg = np.insert(np.array(np.mean(np.array(old_predict_avg), axis=0)), 0, source_class, axis=0)
            out.append(do_predict_avg)
            #old_out.append(old_predict_avg)
            #print('number of samples:{}'.format(total_num_samples))
    out = np.sum(np.array(out), axis=1)
    out[potential_target] = 0
    np.savetxt(args.output_dir + "/test_sum_" + "t" + str(potential_target) + ".txt",
               out, fmt="%s")
    idx = np.arange(0, len(out), 1, dtype=int)
    out = np.insert(out[:, np.newaxis], 0, idx, axis=1)

    ind = np.argsort(out[:, 1])[::-1]
    flag_list = out[ind][0][0]

    return flag_list



def analyze_source_class2(model, model_name, target_class, potential_target, num_class, ana_layer, num_sample, th=3):
    out = []
    old_out = []
    common_out = []
    for source_class in range(0, num_class):
        #print('analyzing source class: {}'.format(source_class))
        #class_loader = get_custom_class_loader(args.data_set, args.batch_size, source_class, args.data_name, target_class)
        for cur_layer in ana_layer:
            # load sensitive neuron
            hidden_test = np.loadtxt(
                args.output_dir + "/test_pre0_" + "c" + str(source_class) + "_layer_" + str(cur_layer) + ".txt")
            # check common important neuron
            temp = hidden_test[:, [0, (potential_target + 1)]]
            ind = np.argsort(temp[:, 1])[::-1]
            temp = temp[ind]

            # find outlier hidden neurons
            top_num = int(len(outlier_detection(temp[:, 1], max(temp[:, 1]), th=max(2, args.confidence2), verbose=False)))
            top_neuron = list(temp[:top_num].T[0].astype(int))
            np.savetxt(args.output_dir + "/outstanding_" + "c" + str(source_class) + "_target_" + str(potential_target) + ".txt",
                       temp[:,0].astype(int), fmt="%s")
            #print('significant neuron: {}'.format(top_num))
            #'''
            # get source to source top neuron
            temp_s = hidden_test[:, [0, (source_class + 1)]]
            ind = np.argsort(temp_s[:, 1])[::-1]
            temp_s = temp_s[ind]

            # find outlier hidden neurons
            top_num_s = int(len(outlier_detection(temp_s[:, 1], max(temp_s[:, 1]), th=args.confidence2, verbose=False)))
            top_neuron_s = list(temp_s[:top_num_s].T[0].astype(int))

            len_top_s = max(len(top_neuron), top_num_s)
            top_neuron_s = temp_s[:len_top_s]

            common = np.intersect1d(top_neuron, top_neuron_s)
            if source_class == potential_target:
                common = []
            #print('source {}, common {}, num_tar {}, num_src {}, ratio {}'.format(
            #    source_class, len(common), len(top_neuron), len(top_neuron_s), len(common) / len(top_neuron)))
            common_out.append(len(common))
            #ca = Counter(top_neuron)
            #cb= Counter(top_neuron_s)
            #diff = sorted((ca - cb).elements())
            #print('significant neuron: {}, fraction: {}'.format(len(diff), len(diff)/top_num))
            #'''
            #top_neuron = diff
            #np.savetxt(args.output_dir + "/sensitive" + "c" + str(source_class) + "_target_" + str(potential_target) + ".txt",
            #           top_neuron, fmt="%s")
            #top_neuron = [24,429,297,401,96,459,246,367,91,509,445,287,320,291,182,198,474,47,308,113,253,290,276,476,73,220,505,105,144,410,319,141,212,15,81,5,275,448,185,89,337,173,1,214,493,176,12,265,458,87,322,331,56,384,400,54,145,243,97,51,109,510,465,369,83,330,126,497,292,157,324,247,484,499,306,372,390,427,127,295,16,354,230,72,86,371,332,422,502,67,500,356,115,314,99,231,450,368,187,441,211,340,169,472,263,155,160,238,192,71,226]
            #prepare mask
            '''
            mask = np.zeros(len(temp))
            mask[top_neuron] = 1
            mask = torch.from_numpy(mask).to(device)

            model1, model2 = split_model(model, model_name, split_layer=cur_layer)
            model1.eval()
            model2.eval()

            do_predict_avg = []
            old_predict_avg = []
            total_num_samples = 0
            for image, gt in class_loader:
                if total_num_samples >= num_sample:
                    break

                image, gt = image.to(device), gt.to(device)

                # compute output
                with torch.no_grad():
                    dense_output = model1(image)
                    # dense_output = dense_output.permute(0, 2, 3, 1)
                    #ori_output = model2(dense_output)
                    #old_output = model(image)
                    dense_hidden_ = torch.clone(torch.reshape(dense_output, (dense_output.shape[0], -1)))
                    # ori_output = filter_model(image)
                    do_predict_neu = []
                    do_predict = []
                    # do convention for each neuron

                    hidden_do = (mask * dense_hidden_).cpu().detach().numpy()
                    #dense_output_ = torch.clone(dense_hidden_)
                    #dense_output_ = dense_output_ + hidden_do
                    #dense_output_ = torch.reshape(dense_output_, dense_output.shape)
                    #dense_output_ = dense_output_.to(device)
                    #output_do = model2(dense_output_.float()).cpu().detach().numpy()
                    do_predict = np.mean(np.array(hidden_do), axis=0)
                    #old_predict = np.mean(old_output.cpu().detach().numpy(), axis=0)

                do_predict_avg.append(do_predict)  # batchx4096x11
                #old_predict_avg.append(old_predict)
                total_num_samples += len(gt)
            # average of all baches
            do_predict_avg = np.mean(np.array(do_predict_avg), axis=0)  # 4096x10
            #do_predict_avg = np.insert(np.array(do_predict_avg), 0, source_class, axis=0)
            #old_predict_avg = np.insert(np.array(np.mean(np.array(old_predict_avg), axis=0)), 0, source_class, axis=0)
            out.append(do_predict_avg)
            #old_out.append(old_predict_avg)
            #print('number of samples:{}'.format(total_num_samples))
            '''
    '''        
    out = np.sum(np.array(out), axis=1)
    out[potential_target] = 0
    np.savetxt(args.output_dir + "/test_sum_" + "t" + str(potential_target) + ".txt",
               out, fmt="%s")
    idx = np.arange(0, len(out), 1, dtype=int)
    out = np.insert(out[:, np.newaxis], 0, idx, axis=1)

    ind = np.argsort(out[:, 1])[::-1]
    flag_list = out[ind][0][0]
    '''
    common_out = np.argsort(common_out)
    flag_list = common_out[-1]
    return flag_list


def solve_detect_common_outstanding_neuron(num_class, ana_layer):
        '''
        find common outstanding neurons
        return potential attack base class and target class
        '''
        #print('Detecting common outstanding neurons.')

        flag_list = []
        top_list = []
        top_neuron = []

        for each_class in range (0, num_class):
            top_list_i, top_neuron_i = detect_eachclass_all_layer(each_class, num_class, ana_layer)
            top_list = top_list + top_list_i
            top_neuron.append(top_neuron_i)
            #self.plot_eachclass_expand(each_class)

        #top_list dimension: 10 x 10 = 100
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
            #l = np.ones(len(hidden_test_)) * cur_layer
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

        # top_list x10
        # find outlier
        #flag_list = self.outlier_detection(top_list, top_num, cur_class)

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
        #top_ = self.find_outstanding_neuron(each_class, prefix="all_")
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

    #'''
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
    #'''
    hidden_test = []
    for cur_layer in ana_layer:
        #hidden_test_ = np.loadtxt(RESULT_DIR + prefix + "test_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt")
        hidden_test_ = np.loadtxt(args.output_dir + '/' + prefix + "test_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt")
        #l = np.ones(len(hidden_test_)) * cur_layer
        hidden_test_ = np.insert(np.array(hidden_test_), 0, cur_layer, axis=1)
        hidden_test = hidden_test + list(hidden_test_)
    '''
    hidden_test = np.loadtxt(RESULT_DIR + prefix + "test_pre0_"  + "c" + str(cur_class) + "_layer_13" + ".txt")
    '''
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

        #print('median: %f, MAD: %f' % (median, mad))
        #print('anomaly index: %f' % min_mad)

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
            #if i >= self.num_target:
            #    break

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

    for i, (images, labels) in enumerate(data_loader):
        for idx, (images_adv, labels_adv) in enumerate(adv_loader):
            _input = torch.cat((images[:44], images_adv[:20]), 0)
            _output = torch.cat((labels[:44], labels_adv[:20]), 0)
            images = _input
            labels = _output
        labels = labels.long()
        images, labels = images.to(device), labels.to(device)
        target = (torch.ones(20, dtype=torch.int64) * target_class).to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels) - reg * criterion(output[-20:], target)

        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()

        loss.backward()
        optimizer.step()


    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


def train_trigger(model, criterion, optimizer, target_class, image, batch_size, reg=0.9):
    model.module.generator.train()
    model.module.target_model.eval()
    total_correct = 0
    total_loss = 0.0

    # images
    images = image.repeat(batch_size, 1, 1, 1)
    images = images.to(device)

    target = (torch.ones(images.shape[0], dtype=torch.int64) * target_class).to(device)

    optimizer.zero_grad()
    output = model(images)
    #K.mean(model1(input_img)[:, output_index]) - reg * K.mean(K.square(input_img))
    uap = torch.unsqueeze(model.module.generator.uap, dim=0)
    loss = criterion(output, target)# + reg * torch.mean(torch.square(uap + image.to(device)))
    #loss = criterion(output, target)

    pred = output.data.max(1)[1]
    total_correct += pred.eq(target.view_as(pred)).sum()
    total_loss += loss.item()

    loss.backward()
    optimizer.step()

    model.module.generator.uap.data = torch.clamp(model.module.generator.uap.data, -0.1, 0.1)

    return total_loss, float(total_correct)


def _adjust_learning_rate(optimizer, epoch, lr):
    if epoch < 21:
        lr = lr
    elif epoch < 100:
        lr = 0.1 * lr
    else:
        lr = 0.0009
    #print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
        pre_analysis()

