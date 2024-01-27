#encoding:utf-8
import os, sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
sys.path.append(os.getcwd())
import argparse
from tqdm import tqdm
import pandas as pd

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torch.utils.data import TensorDataset, DataLoader

import arguments
import utils
# import ForTrsutils as utils
# import arguments, OriginalUtils
from models.ensemble import Ensemble


def test(model, datafolder, return_acc=False):
    inputs = torch.load(os.path.join(datafolder, 'inputs.pt')).cpu()
    labels = torch.load(os.path.join(datafolder, 'labels.pt')).cpu()

    testset = TensorDataset(inputs, labels)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    correct = []
    with torch.no_grad():
        for (x, y) in testloader:
            x, y = x.cuda(), y.cuda()
            outputs = model(x)
            _, preds = outputs.max(1)
            correct.append(preds.eq(y))
    correct = torch.cat(correct, dim=0)
    if return_acc:
        return 100. * correct.sum().item() / len(testset)
    else:
        return correct


def get_args():
    parser = argparse.ArgumentParser(description='Evaluation of Black-box Transfer Robustness of Ensembles',
                                     add_help=True)
    arguments.model_args(parser)
    arguments.data_args(parser)
    arguments.bbox_eval_args(parser)
    args = parser.parse_args()
    return args


def main():
    # get args
    args = get_args()

    # set up gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    assert torch.cuda.is_available()

    # load models
    if 'gal' in args.model_file:
        leaky_relu = True
    else:
        leaky_relu = False

    #ensemble = utils.get_models(args, train=False, as_ensemble=True, model_file=args.model_file, leaky_relu=leaky_relu)
    # ensemble = utils.get_models(args, train=False, as_ensemble=True, model_file=args.model_file, leaky_relu=leaky_relu,is_paral='trs' in args.model_file)
    ensemble = utils.get_models(args, train=False, as_ensemble=True, model_file=args.model_file, leaky_relu=leaky_relu,is_paral=False)

    train_seed = args.model_file.split('/')[-3]
    train_alg = args.model_file.split('/')[-4]

    eps_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
    loss_fn_list = ['xent', 'cw']
    surrogate_model_list = ['{:s}{:d}'.format(args.which_ensemble, i) for i in [3, 5, 8]]
    method_list = ['fgsm_steps_{:d}'.format(1),'dim_0.5_steps_{:d}'.format(100), 'sgm_0.2_steps_{:d}'.format(100)]
    # index_list = ['{:s}_{:s}_mpgd'.format(a, b) for a in surrogate_model_list for b in loss_fn_list]
    # index_list += ['{:s}_{:s}_{:s}'.format(a, b, c) for a in surrogate_model_list for b in loss_fn_list for c in
    #                method_list]
    # index_list.append('all')
    index_list = ['mpgd_steps_10','mpgd_steps_100','fgsm','mdi2','sgm','all']

    #以上是csv保存的顺序mpgd(3,5,8),然后是fgsm,dim,sgm
    random_start = 3
    filepath = 'data/transfer_adv_examples/eps_0.01'
    name = os.listdir(filepath)
    input_mpgd_steps_10_list = []
    input_mpgd_steps_100_list = []
    input_fgsm_list = []
    input_sgm_list = []
    input_mdi2_list = []
    #input_sa_list = []

    count_mpgd_10 = 0
    count_mpgd_100 = 0
    count_fgsm = 0
    count_mdi2 = 0
    count_sgm = 0
    #count_sa = 0
    for input_filename in name:
        if 'mpgd_steps_10_' in input_filename:
            count_mpgd_10 = count_mpgd_10 + 1
            input_mpgd_steps_10_list.append(input_filename)
        elif 'mpgd_steps_100_' in input_filename:
            count_mpgd_100 = count_mpgd_100 + 1
            input_mpgd_steps_100_list.append(input_filename)
        elif 'fgsm' in input_filename:
            count_fgsm = count_fgsm + 1
            input_fgsm_list.append(input_filename)
        elif 'mdi2' in input_filename:
            count_mdi2 = count_mdi2 + 1
            input_mdi2_list.append(input_filename)
        elif 'sgm' in input_filename:
            count_sgm = count_sgm + 1
            input_sgm_list.append(input_filename)
        # elif 'sa' in input_filename:
        #     count_sa = count_sa + 1
        #     input_sa_list.append(input_filename)
    #找出需要进行黑盒测试的对抗样例
    #print(count_mpgd_10, count_mpgd_100, count_fgsm, count_dim, count_sgm)
    #input_list =[]
    # input_list = ['from_{:s}_{:s}_mpgd_steps_{:d}'.format(a, b, 100) for a in surrogate_model_list for b in
    #               loss_fn_list]
    # input_list += ['from_{:s}_{:s}_mpgd_steps_{:d}'.format(a, b, 10) for a in surrogate_model_list for b in
    #               loss_fn_list]
    # input_list += ['from_{:s}_{:s}_{:s}'.format(a, b, c) for a in surrogate_model_list for b in loss_fn_list for c in
    #                method_list]

    #print("index_list:", len(index_list), index_list)
    #print("input_mpgd_steps_10_list:", len(input_mpgd_steps_10_list), input_mpgd_steps_10_list)
    #print("input_mpgd_steps_100_list:", len(input_mpgd_steps_100_list), input_mpgd_steps_100_list)
    #print("input_fgsm_list:", len(input_fgsm_list), input_fgsm_list)
    #print("input_mdi2_list:", len(input_mdi2_list), input_mdi2_list)
    #print("input_sgm_list:", len(input_sgm_list), input_sgm_list)
    #print("input_sa_list:", len(input_sa_list), input_sa_list)
    rob = {}
    rob['source'] = index_list
    acc_list = [[] for _ in range(len(eps_list))]

    data_root = os.path.join(args.data_dir, args.folder)

    # clean acc
    input_folder = os.path.join(data_root, 'clean')
    clean_acc = test(ensemble, input_folder, return_acc=True)
    clean_acc_list = [clean_acc for _ in range(6)]#acc的长度，就是一共有几行
    rob['clean'] = clean_acc_list
    #print(len(rob['clean']))

    # transfer attacks
    for i, eps in enumerate(tqdm(eps_list, desc='eps', leave=True, position=0)):
        input_folder = os.path.join(data_root, 'eps_{:.2f}'.format(eps))
        correct_over_input = []
        #找出各个黑盒攻击算法的准确率：
        #mpgd_10
        correct_over_mpgd_10_rs = []
        for input_mpgd_steps_10 in tqdm(input_mpgd_steps_10_list, desc='source', leave=False, position=1):
            datafolder = os.path.join(input_folder, input_mpgd_steps_10)
            correct_over_mpgd_10_rs.append(test(ensemble, datafolder))

        correct_over_mpgd_10_rs = torch.stack(correct_over_mpgd_10_rs, dim=-1).all(dim=-1)
        acc_list[i].append(100. * correct_over_mpgd_10_rs.sum().item() / len(correct_over_mpgd_10_rs))
        correct_over_input.append(correct_over_mpgd_10_rs)

        tqdm.write('Clean acc: {:.2f}%, eps: {:.2f}, transfer from {:s} acc: {:.2f}%'.format(
            clean_acc, eps, 'mpdg_10', 100. * correct_over_mpgd_10_rs.sum().item() / len(correct_over_mpgd_10_rs)
        ))
        # mpgd_100
        correct_over_mpgd_100_rs = []
        for input_mpgd_steps_100 in tqdm(input_mpgd_steps_100_list, desc='source', leave=False, position=1):
            datafolder = os.path.join(input_folder, input_mpgd_steps_100)
            correct_over_mpgd_100_rs.append(test(ensemble, datafolder))

        correct_over_mpgd_100_rs = torch.stack(correct_over_mpgd_100_rs, dim=-1).all(dim=-1)
        acc_list[i].append(100. * correct_over_mpgd_100_rs.sum().item() / len(correct_over_mpgd_100_rs))
        correct_over_input.append(correct_over_mpgd_100_rs)

        tqdm.write('Clean acc: {:.2f}%, eps: {:.2f}, transfer from {:s} acc: {:.2f}%'.format(
            clean_acc, eps, 'mpdg_100', 100. * correct_over_mpgd_100_rs.sum().item() / len(correct_over_mpgd_100_rs)
        ))
        #fgsm
        correct_over_fgsm_rs = []
        for input_fgsm in tqdm(input_fgsm_list, desc='source', leave=False, position=1):
            datafolder = os.path.join(input_folder, input_fgsm)
            correct_over_fgsm_rs.append(test(ensemble, datafolder))

        correct_over_fgsm_rs = torch.stack(correct_over_fgsm_rs, dim=-1).all(dim=-1)
        acc_list[i].append(100. * correct_over_fgsm_rs.sum().item() / len(correct_over_fgsm_rs))
        correct_over_input.append(correct_over_fgsm_rs)

        tqdm.write('Clean acc: {:.2f}%, eps: {:.2f}, transfer from {:s} acc: {:.2f}%'.format(
            clean_acc, eps, 'fgsm', 100. * correct_over_fgsm_rs.sum().item() / len(correct_over_fgsm_rs)
        ))
        # mdi2
        correct_over_mdi2_rs = []
        for input_mdi2 in tqdm(input_mdi2_list, desc='source', leave=False, position=1):
            datafolder = os.path.join(input_folder, input_mdi2)
            correct_over_mdi2_rs.append(test(ensemble, datafolder))

        correct_over_mdi2_rs = torch.stack(correct_over_mdi2_rs, dim=-1).all(dim=-1)
        acc_list[i].append(100. * correct_over_mdi2_rs.sum().item() / len(correct_over_mdi2_rs))
        correct_over_input.append(correct_over_mdi2_rs)

        tqdm.write('Clean acc: {:.2f}%, eps: {:.2f}, transfer from {:s} acc: {:.2f}%'.format(
            clean_acc, eps, 'mdi2', 100. * correct_over_mdi2_rs.sum().item() / len(correct_over_mdi2_rs)
        ))
        # sgm
        correct_over_sgm_rs = []
        for input_sgm in tqdm(input_sgm_list, desc='source', leave=False, position=1):
            datafolder = os.path.join(input_folder, input_sgm)
            correct_over_sgm_rs.append(test(ensemble, datafolder))

        correct_over_sgm_rs = torch.stack(correct_over_sgm_rs, dim=-1).all(dim=-1)
        acc_list[i].append(100. * correct_over_sgm_rs.sum().item() / len(correct_over_sgm_rs))
        correct_over_input.append(correct_over_sgm_rs)

        tqdm.write('Clean acc: {:.2f}%, eps: {:.2f}, transfer from {:s} acc: {:.2f}%'.format(
            clean_acc, eps, 'sgm', 100. * correct_over_sgm_rs.sum().item() / len(correct_over_sgm_rs)
        ))
        # sa-tim
        # correct_over_sa_rs = []
        # for input_sa in tqdm(input_sa_list, desc='source', leave=False, position=1):
        #     datafolder = os.path.join(input_folder, input_sa)
        #     correct_over_sa_rs.append(test(ensemble, datafolder))
        #
        # correct_over_sa_rs = torch.stack(correct_over_sa_rs, dim=-1).all(dim=-1)
        # acc_list[i].append(100. * correct_over_sa_rs.sum().item() / len(correct_over_sa_rs))
        # correct_over_input.append(correct_over_sa_rs)
        #
        # tqdm.write('Clean acc: {:.2f}%, eps: {:.2f}, transfer from {:s} acc: {:.2f}%'.format(
        #     clean_acc, eps, 'sa', 100. * correct_over_sa_rs.sum().item() / len(correct_over_sa_rs)
        # ))



        correct_over_input = torch.stack(correct_over_input, dim=-1).all(dim=-1)
        acc_list[i].append(100. * correct_over_input.sum().item() / len(correct_over_input))

        tqdm.write('Clean acc: {:.2f}%, eps: {:.2f}, transfer acc: {:.2f}%'.format(
            clean_acc, eps, 100. * correct_over_input.sum().item() / len(correct_over_input)
        ))

        rob[str(eps)] = acc_list[i]

    # save to file
    if args.save_to_csv:
        output_root = os.path.join('results', 'bbox', train_alg, train_seed)

        if not os.path.exists(output_root):
            os.makedirs(output_root)
        output_filename = args.model_file.split('/')[-2]
        output = os.path.join(output_root, '.'.join((output_filename, 'csv')))

        df = pd.DataFrame(rob)
        if args.append_out and os.path.isfile(output):
            with open(output, 'a') as f:
                f.write('\n')
            df.to_csv(output, sep=',', mode='a', header=False, index=False, float_format='%.2f')
        else:
            df.to_csv(output, sep=',', index=False, float_format='%.2f')


if __name__ == '__main__':
    main()
