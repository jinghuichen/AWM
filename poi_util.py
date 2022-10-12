import numpy as np
import random
import imageio
import torch.nn as nn
import torch
import torch.nn.functional as F
import os

def patching(clean_sample, attack, pert=None, dataset_nm = 'CIFAR'):
    '''
    this code conducts a patching procedure to generate backdoor data
    **please make sure the input sample's label is different from the target label
    clean_sample: clean input
    '''
    output = np.copy(clean_sample)
    try:
        if attack == 'badnets':
            pat_size = 4
            output[32 - 1 - pat_size:32 - 1, 32 - 1 - pat_size:32 - 1, :] = 1
        else:
            trimg = imageio.imread('./triggers/' + attack + '.png')/255
            if attack == 'l0_inv':
                mask = 1 - np.transpose(np.load('./triggers/mask.npy'), (1, 2, 0))
                output = clean_sample * mask + trimg
            else:
                output = clean_sample + trimg
        output[output < 0] = 0
        output[output > 1] = 1
        return output
    except:
        if attack == 'badnets':
            pat_size = 4
            output[32 - 1 - pat_size:32 - 1, 32 - 1 - pat_size:32 - 1, :] = 1
        else:
            trimg = imageio.imread('./triggers/' + attack + '.png')/255
            if attack == 'l0_inv':
                mask = 1 - np.transpose(np.load('./triggers/mask.npy'), (1, 2, 0))
                output = clean_sample * mask + trimg
            else:
                output = clean_sample + trimg
        output[output < 0] = 0
        output[output > 1] = 1
        return output


def poison_dataset(dataset, label, attack, target_lab=6, portion=0.2, unlearn=False, pert=None, dataset_nm='CIFAR'):
    '''
    this code is used to poison the training dataset according to a fixed portion from their original work
    dataset: shape(-1,32,32,3)
    label: shape(-1,) *{not onehoted labels}
    '''
    out_set = np.copy(dataset)
    out_lab = np.copy(label)

    if attack == 'badnets_all2all':
        for i in random.sample(range(0, dataset.shape[0]), int(dataset.shape[0] * portion)):
            out_set[i] = patching(dataset[i], 'badnets')
            out_lab[i] = label[i] + 1
            if dataset_nm == 'CIFAR':
                if out_lab[i] == 10:
                    out_lab[i] = 0
            elif dataset_nm == 'GTSRB':
                if out_lab[i] == 43:
                    out_lab[i] = 0
    elif attack == 'a2a':
        for i in random.sample(range(0, dataset.shape[0]), int(dataset.shape[0] * portion)):
            out_set[i] = patching(dataset[i], 'trojan_sq')
            out_lab[i] = label[i] + 1
            if dataset_nm == 'CIFAR':
                if out_lab[i] == 10:
                    out_lab[i] = 0
            elif dataset_nm == 'GTSRB':
                if out_lab[i] == 43:
                    out_lab[i] = 0                    
    else:
        indexs = list(np.asarray(np.where(label != target_lab))[0])
        samples_idx = random.sample(indexs, int(dataset.shape[0] * portion))
        for i in samples_idx:
            out_set[i] = patching(dataset[i], attack, pert, dataset_nm = dataset_nm)
            assert out_lab[i] != target_lab
            out_lab[i] = target_lab
    if unlearn:
        return out_set, label
    return out_set, out_lab


def patching_test(dataset, label, attack, target_lab=6, adversarial=False, dataset_nm='CIFAR'):
    """
    This code is used to generate an all-poisoned dataset for evaluating the ASR
    """
    out_set = np.copy(dataset)
    out_lab = np.copy(label)
    if attack == 'badnets_all2all':
        for i in range(out_set.shape[0]):
            out_set[i] = patching(dataset[i], 'badnets')
            out_lab[i] = label[i] + 1
            if dataset_nm == 'CIFAR':
                if out_lab[i] == 10:
                    out_lab[i] = 0
            elif dataset_nm == 'GTSRB':
                if out_lab[i] == 43:
                    out_lab[i] = 0
    elif attack == 'a2a':
        for i in range(out_set.shape[0]):
            out_set[i] = patching(dataset[i], 'trojan_sq')
            out_lab[i] = label[i] + 1
            if dataset_nm == 'CIFAR':
                if out_lab[i] == 10:
                    out_lab[i] = 0
            elif dataset_nm == 'GTSRB':
                if out_lab[i] == 43:
                    out_lab[i] = 0                               
    else:
        for i in range(out_set.shape[0]):
            out_set[i] = patching(dataset[i], attack, dataset_nm = dataset_nm)
            out_lab[i] = target_lab
    if adversarial:
        return out_set, label
    return out_set, out_lab

def patching_test_wanet(dataset, label, mode, target_label, device, ckpt_path):
    out_set = torch.transpose(torch.Tensor(dataset), 1, 3).to(device) # swap dim1 and dim3
    out_lab = torch.Tensor(label).to(device)
    bs = dataset.shape[0] # num of samples
    # num of channels
    input_height = 32
    grid_rescale = 1
    s = 0.5
    k = 4

    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path)
        best_clean_acc = state_dict["best_clean_acc"]
        best_bd_acc = state_dict["best_bd_acc"]
        best_cross_acc = state_dict["best_cross_acc"]
        epoch_current = state_dict["epoch_current"]        
        identity_grid = state_dict["identity_grid"]
        noise_grid = state_dict["noise_grid"]
    print("best_clean_acc | ", best_clean_acc, " | best_cross_acc | ", best_cross_acc, " | best_bd_acc | ", best_bd_acc, " | epoch_current | ", epoch_current)

    # Evaluate Backdoor
    grid_temps = (identity_grid + s * noise_grid / input_height) * grid_rescale
    grid_temps = torch.clamp(grid_temps, -1, 1)

    ins = torch.rand(bs, input_height, input_height, 2).to(device) * 2 - 1
    grid_temps2 = grid_temps.repeat(bs, 1, 1, 1) + ins / input_height
    grid_temps2 = torch.clamp(grid_temps2, -1, 1)

    print(out_set.shape, grid_temps.shape)
    out_set = F.grid_sample(out_set, grid_temps.repeat(bs, 1, 1, 1), align_corners=True) #(,3,32,32)
    print(out_set.shape)

    if mode == "all2one":
        out_lab = torch.ones_like(out_lab) * target_label
    if mode == "all2all":
        out_lab = torch.remainder(out_lab + 1, 10) # for cifar10
    
    return out_set, out_lab.long(), state_dict["netC"]

def patching_train(dataset, label, attack, target_lab=6, adversarial=False, dataset_nm='CIFAR'):
    poison_rate = 0.05
    #poison_rate = 0.20
    #poison_rate = 0.50

    out_set = np.copy(dataset)
    out_lab = np.copy(label)
    
    poison_cand = [i for i in range(dataset.shape[0]) if label[i] != target_lab]
    poison_num = int(poison_rate * len(poison_cand))
    choices = np.random.choice(poison_cand, poison_num, replace=False)

    for i in choices:
        if attack == 'a2a':
            out_set[i] = patching(dataset[i], 'trojan_sq', dataset_nm = dataset_nm)
            out_lab[i] = label[i] + 1
            if dataset_nm == 'CIFAR':
                if out_lab[i] == 10:
                    out_lab[i] = 0
            elif dataset_nm == 'GTSRB':
                if out_lab[i] == 43:
                    out_lab[i] = 0                      
        else:        
            out_set[i] = patching(dataset[i], attack, dataset_nm = dataset_nm)
            out_lab[i] = target_lab
    return out_set, out_lab