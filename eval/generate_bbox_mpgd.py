#encoding:utf-8
import os, sys

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

sys.path.append(os.getcwd())
import argparse, random
from tqdm import tqdm
import pandas as pd

import numpy as np
import torch

# torch.cuda.device_count()

# print(torch.cuda.get_device_name())
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from advertorch.attacks import LinfBasicIterativeAttack, LinfPGDAttack, MomentumIterativeAttack
from advertorch.attacks.utils import multiple_mini_batch_attack, attack_whole_dataset
from advertorch.utils import to_one_hot

import arguments
import utils
from models.ensemble import Ensemble
from distillation import Linf_PGD


# https://github.com/BorealisAI/advertorch/blob/master/advertorch/utils.py
class CarliniWagnerLoss(nn.Module):
    """
    Carlini-Wagner Loss: objective function #6.
    Paper: https://arxiv.org/pdf/1608.04644.pdf
    """

    def __init__(self, conf=50.):
        super(CarliniWagnerLoss, self).__init__()
        self.conf = conf

    def forward(self, input, target):
        """
        :param input: pre-softmax/logits.
        :param target: true labels.
        :return: CW loss value.
        """
        num_classes = input.size(1)
        label_mask = to_one_hot(target, num_classes=num_classes).float()
        correct_logit = torch.sum(label_mask * input, dim=1)
        wrong_logit = torch.max((1. - label_mask) * input, dim=1)[0]
        loss = -F.relu(correct_logit - wrong_logit + self.conf).sum()
        return loss


def get_args():
    parser = argparse.ArgumentParser(description='Evaluation of White-box Robustness of Ensembles with Advertorch',
                                     add_help=True)
    arguments.model_args(parser)
    arguments.data_args(parser)
    arguments.gengerate_bbox_mpgd_args(parser)
    args = parser.parse_args()
    return args


def backward_hook(gamma):
    # implement SGM through grad through ReLU
    def _backward_hook(module, grad_in, grad_out):
        if isinstance(module, nn.ReLU):
            return (gamma * grad_in[0],)

    return _backward_hook


def backward_hook_norm(module, grad_in, grad_out):
    # normalize the gradient to avoid gradient explosion or vanish
    std = torch.std(grad_in[0])
    return (grad_in[0] / std,)


def register_hook_for_resnet(model, arch, gamma):
    # There is only 1 ReLU in Conv module of ResNet-18/34
    # and 2 ReLU in Conv module ResNet-50/101/152
    if arch in ['resnet50', 'resnet101', 'resnet152']:
        gamma = np.power(gamma, 0.5)
    backward_hook_sgm = backward_hook(gamma)

    for name, module in model.named_modules():
        if 'relu' in name and not '0.relu' in name:
            module.register_backward_hook(backward_hook_sgm)

        # e.g., 1.layer1.1, 1.layer4.2, ...
        # if len(name.split('.')) == 3:
        if len(name.split('.')) >= 2 and 'layer' in name.split('.')[-2]:
            module.register_backward_hook(backward_hook_norm)


def generate_adversarial_example(model, data_loader, adversary, output_dir, prob):
    """
    evaluate model by black-box attack
    """
    model.eval()
    all_clean = None
    all_adv = None
    all_label = None

    for batch_idx, (inputs, true_class) in enumerate(data_loader):
        inputs, true_class = inputs.cuda(), true_class.cuda()

        # SGM & mPGD attack
        inputs_adv = adversary.perturb(inputs, true_class)
        # M-DI2-FGSM
        # print("using MDI2")
        # inputs_adv = adversary.perturb(input_diversity(inputs, 64, 66, prob), true_class)

        all_adv = inputs_adv if all_adv is None else torch.cat((all_adv, inputs_adv), 0)
        all_label = true_class if all_label is None else torch.cat((all_label, true_class), 0)
        all_clean = inputs if all_clean is None else torch.cat((all_clean, inputs), 0)

        # save_images(inputs_adv.detach().cpu().numpy(), img_list=img_path,
        #             idx=idx, output_dir=args.output_dir)
        # assert False
        # if batch_idx % args.print_freq == 0:
        #     print('generating: [{0}/{1}]'.format(batch_idx, len(data_loader)))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    torch.save(all_adv, os.path.join(output_dir, 'inputs.pt'))
    torch.save(all_label, os.path.join(output_dir, 'labels.pt'))
    clean_path = os.path.join(output_dir[:output_dir.rfind('/') - 9], 'clean')
    if not os.path.exists(clean_path):
        os.makedirs(clean_path)
        torch.save(all_clean, os.path.join(clean_path, 'inputs.pt'))
        torch.save(all_label, os.path.join(clean_path, 'labels.pt'))


def input_diversity(input_tensor, image_width, image_resize, prob):
    rnd = torch.FloatTensor(1,).uniform_(image_width, image_resize)[0]
    rnd = rnd.floor().int()
    rescaled = F.interpolate(input_tensor, [rnd, rnd], mode="nearest")
    h_rem = image_resize - rnd
    w_rem = image_resize - rnd
    pad_top = torch.FloatTensor(1,).uniform_(0, h_rem + 1)[0]
    pad_top = pad_top.floor().int()
    pad_bottom = h_rem - pad_top
    pad_left = torch.FloatTensor(1, ).uniform_(0, w_rem + 1)[0]
    pad_left = pad_left.floor().int()
    pad_right = w_rem - pad_left
    padded = F.pad(rescaled, [pad_left, pad_right, pad_top, pad_bottom], mode='constant', value=0.0)
    padded = padded.view((input_tensor.shape[0], 3, image_resize, image_resize))
    padded = F.interpolate(padded, [image_width, image_width])
    if torch.FloatTensor(1,).uniform_(0, 1)[0] < torch.tensor(prob):
        return padded
    return input_tensor


def main():
    # get args
    args = get_args()
    print(args)
    torch.cuda.set_device(0)
    print("in function get_args :model_num:   " + str(args.model_num)+"model_file:   "+str(args.model_file))

    # set up gpus

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # print("gpu:"+str(args.gpu)+"is using,generating bbox adversarial examples!")
    # assert torch.cuda.is_available()

    # load models
    if 'gal' in args.model_file:
        leaky_relu = True
    else:
        leaky_relu = False
    ensemble = utils.get_models(args, train=False, as_ensemble=True, model_file=args.model_file, leaky_relu=leaky_relu)
    # ensemble = Ensemble([ensemble.models[0]])

    # SGM Attack
    # print("using SGM attack!")
    # if args.gamma < 1.0:
    #     print("using sgm gamma=0.2 to generate adversarial examples!")
    #     for model in ensemble.models:
    #         register_hook_for_resnet(model, arch=args.arch, gamma=args.gamma)


    #to this is sgm attack

    # get data loaders
    total_sample_num = 10000
    if args.subset_num:
        random.seed(0)
        subset_idx = random.sample(range(total_sample_num), args.subset_num)
        testloader = utils.get_testloader(args, batch_size=200, shuffle=False, subset_idx=subset_idx)
    else:
        testloader = utils.get_testloader(args, batch_size=200, shuffle=False)

    eps_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
    loss_fn_list = ['cw','xent']

    for loss_fn_l in tqdm(loss_fn_list, desc='loss function', leave=True, position=0):
        loss_fn = nn.CrossEntropyLoss(reduction="sum") if loss_fn_l == 'xent' else CarliniWagnerLoss(args.cw_conf)

        for eps in tqdm(eps_list, desc='PGD eps', leave=False, position=1):
            # correct_or_not = []

            # Random start
            for i in tqdm(range(args.random_start), desc='Random Start', leave=False, position=1):
                torch.manual_seed(i)

                test_iter = tqdm(testloader, desc='Batch', leave=False, position=2)

                if args.momentum > 0.0:
                    print("using momentum:"+str(args.momentum)+" adversary !")
                    adversary = MomentumIterativeAttack(predict=ensemble, loss_fn=loss_fn,
                                                        eps=eps, nb_iter=args.num_steps, eps_iter=eps / 5,
                                                        decay_factor=args.momentum,
                                                        clip_min=0.0, clip_max=1.0, targeted=False)
                else:
                    adversary = LinfPGDAttack(predict=ensemble, loss_fn=loss_fn,
                                              eps=eps, nb_iter=args.num_steps, eps_iter=eps / 5,
                                              rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
                    #when fgsm
                    # adversary = LinfPGDAttack(predict=ensemble, loss_fn=loss_fn,
                    #                           eps=eps, nb_iter=args.num_steps, eps_iter=eps,
                    #                           rand_init=False, clip_min=0.0, clip_max=1.0, targeted=False)

                #对抗样例存放地址
                cur_out_dir = os.path.join(args.data_dir, args.output_dir, 'eps_{:.2f}'.format(eps),
                                          'from_baseline{:d}_{:s}_mpgd_steps_{:d}_{:d}'.format(len(ensemble.models),
                                                                                                 loss_fn_l,
                                                                                                 args.num_steps,
                                                                                                 i))

                #generate sgm adversarial examples do not use momentum 注意注释掉argumnets 里的momentum
                # cur_out_dir = os.path.join(args.data_dir, args.output_dir, 'eps_{:.2f}'.format(eps),
                #                           'from_baseline{:d}_{:s}_sgm_0.2_steps_{:d}'.format(len(ensemble.models),
                #                                                                                loss_fn_l,
                #                                                                                args.num_steps
                #                                                                                ))
                #gengerate mdi2 adversarial examples
                # cur_out_dir = os.path.join(args.data_dir, args.output_dir, 'eps_{:.2f}'.format(eps),
                #                             'from_baseline{:d}_{:s}_mdi2_0.5_steps_{:d}'.format(len(ensemble.models),
                #                                                                                  loss_fn_l,
                #                                                                                  args.num_steps
                #                                                                                  ))
                # gengerate fgsm adversarial examples
                # cur_out_dir = os.path.join(args.data_dir, args.output_dir, 'eps_{:.2f}'.format(eps),
                #                             'from_baseline{:d}_{:s}_fgsm_steps_{:d}'.format(len(ensemble.models),
                #                                                                                  loss_fn_l,
                #                                                                                  args.num_steps
                #                                                                                  ))

                generate_adversarial_example(model=ensemble, data_loader=test_iter,
                                             adversary=adversary, output_dir=cur_out_dir, prob=args.mdi_prob)


if __name__ == '__main__':
    main()
