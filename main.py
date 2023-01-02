import os
from re import L
import time
import argparse
from tkinter import W
import numpy as np
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torch.nn.functional as F

import models
import torch.nn as nn
from poi_util import patching_test, patching_test_wanet
import h5py
import random
import data.poison_cifar as poison


parser = argparse.ArgumentParser(description='Train poisoned networks')

# Basic model parameters.
parser.add_argument('--arch', type=str, default='resnet18',
                    choices=['resnet18', 'MobileNetV2', 'vgg19_bn', 'small_vgg', 'ShuffleNetV2'])
#parser.add_argument('--checkpoint', type=str, required=True, help='The checkpoint direc')
parser.add_argument('--batch-size', type=int, default=128, help='the batch size for dataloader')
parser.add_argument('--outer', type=int, default=20, help='the number of outer optimization epochs')
parser.add_argument('--inner', type=int, default=5, help='do inner optimization for several epochs')
parser.add_argument('--data-dir', type=str, default='./data', help='dir to the dataset')
parser.add_argument('--output-dir', type=str, default='logs/models/')

parser.add_argument('--trigger-info', type=str, default='', help='The information of backdoor trigger')
parser.add_argument('--poison-target', type=int, default=0, help='target class of backdoor attack')
parser.add_argument('--trigger-alpha', type=float, default=1.0, help='the transparency of the trigger pattern.')
parser.add_argument('--alpha', type=float, default=0.9)
parser.add_argument('--gamma', type=float, default=1e-8)

parser.add_argument('--trigger-norm', type=float, default=1000)
parser.add_argument('--shrink-steps', type=int, default=0)
parser.add_argument('--lr1', type=float, default=1e-3, help='the learning rate for shrinking step')
parser.add_argument('--lr2', type=float, default=1e-2, help='the learning rate for washing step')
parser.add_argument('--samples', type=int, default=500)

parser.add_argument('--attack', type=str, default='badnets', choices=['badnets','trojan-sq','trojan-wm','l0-inv','l2-inv','a2a', 'clb', 'blend', 'wanet'])
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10','gtsrb'])


args = parser.parse_args()
args_dict = vars(args)
os.makedirs(args.output_dir, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#from datafree.utils import ImagePool
#data_pool1 = ImagePool(root='trigger_beforeshrink')
#data_pool2 = ImagePool(root='trigger_aftershrink')

def main():
    data_transforms = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip()
    ])    

    transform_train = transforms.Compose([
        transforms.ToTensor()
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
    tsf = transforms.ToTensor()

    if args.dataset == 'cifar10':
        args.num_classes = 10
        trainset = CIFAR10(root=args.data_dir, train=True, download=True, transform=None)
        testset = CIFAR10(root=args.data_dir, train=False, download=True, transform=None)
        x_train, y_train = trainset.data, trainset.targets
        x_test, y_test = testset.data, testset.targets
        x_train = x_train.astype('float32')/255
        x_test = x_test.astype('float32')/255 #->(0,1)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
    else: # gtsrb
        args.num_classes = 43
        f = h5py.File('./data/gtsrb_dataset.h5','r')
        x_train = np.asarray(f['X_train'])/255
        x_test = np.asarray(f['X_test'])/255
        y_train = np.argmax(np.asarray(f['Y_train']), axis=1)
        y_test = np.argmax(np.asarray(f['Y_test']), axis=1)
        randidx = np.arange(y_test.shape[0])
        np.random.shuffle(randidx)
        x_test = x_test[randidx]
        y_test = y_test[randidx]

    if args.attack == 'badnets':
        x_poi_test, y_poi_test= patching_test(x_test, y_test, 'badnets', target_lab=8)
        if args.arch == 'small_vgg':
            args.checkpoint = './checkpoint/'+str(args.dataset)+'_vgg_bd.th'
        else:
            args.checkpoint = './checkpoint/'+str(args.dataset)+'_r18_bd.th'    
    elif args.attack == 'trojan-sq':
        x_poi_test, y_poi_test= patching_test(x_test, y_test, 'trojan_sq', target_lab=2) 
        if args.arch == 'small_vgg':
            args.checkpoint = './checkpoint/'+str(args.dataset)+'_vgg_sq.th'
        else:
            args.checkpoint = './checkpoint/'+str(args.dataset)+'_r18_sq.th'           
    elif args.attack == 'trojan-wm':        
        x_poi_test, y_poi_test= patching_test(x_test, y_test, 'trojan_wm', target_lab=2)
        if args.arch == 'small_vgg':
            args.checkpoint = './checkpoint/'+str(args.dataset)+'_vgg_wm.th'
        else:
            args.checkpoint = './checkpoint/'+str(args.dataset)+'_r18_wm.th'  
    elif args.attack == 'l0-inv':        
        x_poi_test, y_poi_test= patching_test(x_test, y_test, 'l0_inv', target_lab=0)
        if args.arch == 'small_vgg':
            args.checkpoint = './checkpoint/'+str(args.dataset)+'_vgg_l0.th'
        else:
            args.checkpoint = './checkpoint/'+str(args.dataset)+'_r18_l0.th'           
    elif args.attack == 'l2-inv':    
        x_poi_test, y_poi_test= patching_test(x_test, y_test, 'l2_inv', target_lab=0)
        if args.arch == 'small_vgg':
            args.checkpoint = './checkpoint/'+str(args.dataset)+'_vgg_l2.th'
        else:
            args.checkpoint = './checkpoint/'+str(args.dataset)+'_r18_l2.th'
    elif args.attack == 'a2a':
        x_poi_test, y_poi_test= patching_test(x_test, y_test, 'a2a', target_lab=0, adversarial=False, dataset_nm=args.dataset)
        if args.arch == 'small_vgg':
            args.checkpoint = './checkpoint/'+str(args.dataset)+'_vgg_a2a.th'
        else:
            args.checkpoint = './checkpoint/'+str(args.dataset)+'_r18_a2a.th'
    
    y_test = torch.Tensor(y_test.reshape((-1,)).astype(int)).long()
    y_poi_test = torch.Tensor(y_poi_test.reshape((-1,)).astype(int)).long()

    x_test = torch.Tensor(np.transpose(x_test,(0,3,1,2)))
    x_poi_test = torch.Tensor(np.transpose(x_poi_test,(0,3,1,2)))

    num = args.samples
    rand_idx = random.sample(list(np.arange(y_test.shape[0])), int(num))

    ot_idx = [i for i in range(y_test.shape[0]) if i not in rand_idx]

    clean_val = TensorDataset(x_test[rand_idx], y_test[rand_idx])
    poison_test = TensorDataset(x_poi_test[ot_idx], y_poi_test[ot_idx])
    clean_test = TensorDataset(x_test[ot_idx], y_test[ot_idx])


    if args.samples == args.num_classes:
        testset = CIFAR10(root=args.data_dir, train=False, download=True, transform=data_transforms)

        targ = [[] for i in range(args.num_classes)]
        for i in range(10000):
            targ[y_test[i]].append(i)

        #for i in range(args.num_classes):
            #print(y_test[targ[i]])

        indx = [targ[i][1] for i in range(args.num_classes)]

        x_val = x_test[indx]
        y_val = y_test[indx]

        if args.dataset == 'cifar10':
            for j in range(10):
                ind = indx[j]
                y_val = torch.cat([y_val, torch.ones([9]).long()*j], dim = 0)
                for i in range(9):
                    img, _ = testset[ind]
                    img = tsf(img).unsqueeze(dim=0)
                    x_val = torch.cat([x_val, img], dim = 0)
                y_val = y_val.long()

        clean_val = TensorDataset(x_val, y_val)
        del testset


    print(len(clean_val), len(poison_test), len(clean_test))
    args.inner_iters = int(len(clean_val)/args.batch_size)*args.inner

    # we need clean_val clean_test poison_test
    random_sampler = RandomSampler(data_source=clean_val, replacement=True,
                                   num_samples=args.inner_iters * args.batch_size)
    clean_val_loader = DataLoader(clean_val, batch_size=args.batch_size, shuffle=False, sampler=random_sampler, num_workers=0)
    poison_test_loader = DataLoader(poison_test, batch_size=args.batch_size, num_workers=0)
    clean_test_loader = DataLoader(clean_test, batch_size=args.batch_size, num_workers=0)


    # Step 2: load model checkpoints and trigger info
    if args.attack != "wanet":
        state_dict = torch.load(args.checkpoint, map_location=device)
    else:
        state_dict = state_dict_C
    
    if args.arch != "ShuffleNetV2":
        net = getattr(models, args.arch)(num_classes=args.num_classes, norm_layer=nn.BatchNorm2d)
    else:
        net = getattr(models, args.arch)(net_size=0.5, norm_layer=nn.BatchNorm2d)


    load_state_dict(net, orig_state_dict=state_dict)
    net = net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    for name, module in net.named_modules():
        if isinstance(module, models.MaskedConv2d):
            module.include_mask()

    parameters = list(net.named_parameters())
    mask_params = [v for n, v in parameters if "mask" in n]
    mask_names = [n for n, v in parameters if "mask" in n]
    mask_optimizer = torch.optim.Adam(mask_params, lr = args.lr1)

    cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
    po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
    print(po_test_loss, po_test_acc, cl_test_loss, cl_test_acc)

    # Step 3: train backdoored models
        
    # Optional to use shrink_steps to reduce the size of the model
    for i in range(args.shrink_steps):
        start = time.time()
        lr = mask_optimizer.param_groups[0]['lr']
        train_loss, train_acc = shrink(model=net, criterion=criterion, data_loader=clean_val_loader, mask_opt=mask_optimizer)
        cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
        po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
        end = time.time()

    mask_optimizer = torch.optim.Adam(mask_params, lr = args.lr2)
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=mask_optimizer, gamma=0.9)

    # AWM
    for i in range(args.outer):
        start = time.time()
        lr = mask_optimizer.param_groups[0]['lr']
        
        train_loss, train_acc = mask_train(model=net, criterion=criterion, data_loader=clean_val_loader, mask_opt=mask_optimizer)
        cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
        po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
        end = time.time()

        print('Iter \t\t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
        print('EPOCHS {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            (i + 1) * args.inner_iters, po_test_loss, po_test_acc,
            cl_test_loss, cl_test_acc))

        my_lr_scheduler.step()

    torch.save(net.state_dict(), os.path.join(args.output_dir, 'WashedNet.th'))
    vis_mask(net)


def load_state_dict(net, orig_state_dict):
    if 'state_dict' in orig_state_dict.keys():
        orig_state_dict = orig_state_dict['state_dict']
    if "state_dict" in orig_state_dict.keys():
        orig_state_dict = orig_state_dict["state_dict"]

    new_state_dict = OrderedDict()
    for k, v in net.state_dict().items():
        if k in orig_state_dict.keys():
            new_state_dict[k] = orig_state_dict[k]
        elif 'running_mean_noisy' in k or 'running_var_noisy' in k or 'num_batches_tracked_noisy' in k:
            new_state_dict[k] = orig_state_dict[k[:-6]].clone().detach()
        else:
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict)


def clip_mask(model, lower=0.0, upper=1.0):
    params = [param for name, param in model.named_parameters() if 'mask' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)


def include_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, models.MaskedConv2d):
            module.include_noise()


def exclude_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, models.MaskedConv2d):
            module.exclude_noise()


def reset(model, rand_init):
    for name, module in model.named_modules():
        if isinstance(module, models.MaskedConv2d):
            module.reset(rand_init=rand_init, eps=0.4)


def shrink(model, criterion, mask_opt, data_loader):
    model.eval()
    nb_samples = 0

    for i, (images, labels) in enumerate(data_loader):

        images, labels = images.to(device), labels.to(device)
        nb_samples += images.size(0)

        output_clean = model(images)

        loss_nat = criterion(output_clean, labels)
        L1, L2 = Regularization(model)

        loss = args.gamma * L1 + loss_nat

        mask_opt.zero_grad()
        loss.backward()
        mask_opt.step()
        clip_mask(model)
    return 0, 0

def mask_train(model, criterion, mask_opt, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    nb_samples = 0

    batch_pert = torch.zeros([1,3,32,32], requires_grad=True, device=device)

    batch_opt = torch.optim.SGD(params=[batch_pert], lr=10)

    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)

        # step 1: calculate the adversarial perturbation for images
        ori_lab = torch.argmax(model.forward(images),axis = 1).long()
        per_logits = model.forward(images + batch_pert)
        loss = F.cross_entropy(per_logits, ori_lab, reduction='mean')
        loss_regu = torch.mean(-loss)

        batch_opt.zero_grad()
        loss_regu.backward(retain_graph = True)
        batch_opt.step()

    pert = batch_pert * min(1, args.trigger_norm / torch.sum(torch.abs(batch_pert)))
    pert = pert.detach()

    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        nb_samples += images.size(0)
        
        perturbed_images = torch.clamp(images + pert[0], min=0, max=1)
        
        # step 2: calculate noisey loss and clean loss
        mask_opt.zero_grad()
        
        output_noise = model(perturbed_images)
        
        output_clean = model(images)
        pred = torch.argmax(output_clean, axis = 1).long()

        loss_rob = criterion(output_noise, labels)
        loss_nat = criterion(output_clean, labels)
        L1, L2 = Regularization(model)

        print("loss_noise | ", loss_rob.item(), " | loss_clean | ", loss_nat.item(), " | L1 | ", L1.item())
        loss = args.alpha * loss_nat + (1 - args.alpha) * loss_rob + args.gamma * L1

        pred = output_clean.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()
        loss.backward()

        mask_opt.step()
        clip_mask(model)

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / nb_samples
    return loss, acc


def test(model, criterion, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)

    return loss, acc

def pert_test(model, criterion, data_loader, pert):
    model.eval()
    count = np.zeros(args.num_classes)

    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            perturbed_images = torch.clamp(images + pert[0], min=0, max=1)

            output = model(perturbed_images)
            pred = output.data.max(1)[1]

            for j in range(args.num_classes):
                count[j] += pred.eq(j).sum()

    print(count)
    return

def Regularization(model):
    L1=0
    L2=0
    for name, param in model.named_parameters():
        if 'mask' in name:
            L1 += torch.sum(torch.abs(param))
            L2 += torch.norm(param, 2)
    return L1, L2


def vis_mask(model):
    for name, param in model.named_parameters():
        if 'mask' in name:
            box = torch.zeros(10)
            print(name, param.shape)
            for unit in param:
                #data_pool.add(unit)
                unit = unit.view(-1)
                #print(unit)
                for uunit in unit:
                    if uunit != 1:
                        box[int(10*uunit)] = box[int(10*uunit)] + 1
                        #print(uunit)
                    else:
                        box[9] = box[9] + 1
            print(box)


if __name__ == '__main__':
    main()