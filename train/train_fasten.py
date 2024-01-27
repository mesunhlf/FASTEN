import os, sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
sys.path.append(os.getcwd())
import json, argparse, random
from tqdm import tqdm
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import arguments, my_utils
from models.ensemble import Ensemble
# from distillation import Linf_PGD, Linf_distillation
# from target_distillation import *
from untarget_distillation import *

class FAST_Trainer():
    def __init__(self, models, optimizers, schedulers,
                 trainloader, testloader,
                 writer, save_root=None, **kwargs):
        self.models = models
        self.epochs = kwargs['epochs']
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.trainloader = trainloader
        self.testloader = testloader

        self.writer = writer
        self.save_root = save_root
        self.criterion = nn.CrossEntropyLoss()

        # distillation configs
        self.distill_fixed_layer = kwargs['distill_fixed_layer']
        self.beta = kwargs['beta']
        self.distill_eps = kwargs['distill_eps']
        self.distill_eps = kwargs['eps']
        self.distill_cfg = {'eps': kwargs['distill_eps'],
                            'alpha': kwargs['distill_alpha'],
                            'steps': kwargs['distill_steps'],
                            'layer': kwargs['distill_layer'],
                            'rand_start': kwargs['distill_rand_start'],
                            'before_relu': True,
                            'momentum': kwargs['distill_momentum']
                            }


        # diversity training configs
        self.plus_adv = kwargs['plus_adv']
        self.coeff = kwargs['adv_coeff']
        if self.plus_adv:
            self.attack_cfg = {'eps': kwargs['eps'],
                               'alpha': kwargs['alpha'],
                               'steps': kwargs['steps'],
                               'is_targeted': False,
                               'rand_start': True
                               }
        self.depth = kwargs['depth']

        data_num = len(self.trainloader.dataset.index)
        self.last_distill_pert = [0 for i in range(data_num)]
        self.last_adv_pert = [0 for i in range(data_num)]

    def get_epoch_iterator(self):
        iterator = tqdm(list(range(1, self.epochs + 1)), total=self.epochs, desc='Epoch',
                        leave=True, position=1)
        return iterator

    def get_batch_iterator(self):
        loader = my_utils.DistillationLoader(self.trainloader, self.trainloader)
        iterator = tqdm(loader, desc='Batch', leave=False, position=2)
        return iterator

    def run(self):
        epoch_iter = self.get_epoch_iterator()
        for epoch in epoch_iter:
            self.train(epoch)
            self.test(epoch)
            self.save(epoch)

    def train(self, epoch):
        for m in self.models:
            m.train()

        if not self.distill_fixed_layer:
            tqdm.write('Randomly choosing a layer for distillation...')
            self.distill_cfg['layer'] = random.randint(1, self.depth)

        losses = [0 for i in range(len(self.models))]
        xcent_losses = [0 for i in range(len(self.models))]
        adv_losses = [0 for i in range(len(self.models))]
        batch_iter = self.get_batch_iterator()

        for batch_idx, (si, sl, si_idx, ti, tl, ti_idx) in enumerate(batch_iter):
            si, sl = si.cuda(), sl.cuda()
            ti, tl = ti.cuda(), tl.cuda()

            if self.plus_adv:
                adv_inputs_list = []

            distilled_data_list = []
            distilled_data_list2 = []
            model_input = []
            model_target = []
            data_num = len(si)
            pre_pert = [0 for i in range(data_num)]
            for k in range(len(pre_pert)):
                data_idx = si_idx[k]
                pre_pert[k] = self.last_distill_pert[data_idx]

            if (epoch > 1):
                pre_pert = torch.stack(pre_pert)

            init = torch.FloatTensor(si.shape).uniform_(-self.distill_eps, self.distill_eps).cuda()
            rnd_list = np.random.choice(len(self.models), 2, replace=False)
            models = []
            idx0 = rnd_list[0]
            idx1 = rnd_list[1]
            models.append(self.models[idx0])
            models.append(self.models[idx1])
            for i, m in enumerate(models):
                temp, temp_pert = Linf_distillation(m, si, init, **self.distill_cfg)
                ratio = torch.FloatTensor((1)).uniform_(0, 1).cuda()
                if (epoch == 1):
                    pre_pert = temp_pert
                    distilled_data_list.append(temp)
                    distilled_data_list2.append(temp)
                else:
                    eps = self.distill_cfg['eps']
                    temp_pert = ratio * temp_pert + (1 - ratio) * pre_pert
                    temp = torch.max(torch.min(si + temp_pert, si + eps), si - eps)
                    temp = torch.clamp(temp, 0., 1.)
                    distilled_data_list.append(temp)

                    pre_pert = temp - si

            for k in range(len(pre_pert)):
                data_idx = si_idx[k]
                self.last_distill_pert[data_idx] = pre_pert[k]

            loss_list = []
            for i, m in enumerate(self.models):
                loss = 0
                inputs = []
                targets = []

                outputs = m(si)
                inputs.append(F.log_softmax(outputs).clone().detach())
                targets.append(F.softmax(outputs).clone().detach())


                if (i in rnd_list):
                    if(i == rnd_list[0]):
                        select_data = distilled_data_list[1]
                    else:
                        select_data = distilled_data_list[0]
                else:
                    rnd = np.random.randint(0, 2)
                    select_data = distilled_data_list[rnd]

                outputs = m(select_data)
                inputs.append(F.log_softmax(outputs))
                targets.append(F.softmax(outputs))
                loss += self.criterion(outputs, sl)
                xcent_losses[i] += loss.item()

                if self.plus_adv:
                    adv_init = init.clone()
                    temp1, temp_pert1 = Linf_PGD(m, select_data, sl, adv_init, **self.attack_cfg)
                    outputs = m(temp1)
                    inputs.append(F.log_softmax(outputs))
                    targets.append(F.softmax(outputs))
                    adv_loss = self.criterion(outputs, sl)
                    loss = loss + self.coeff * adv_loss
                    adv_losses[i] += self.coeff * adv_loss.item()

                mean_inputs = torch.mean(torch.stack(inputs), dim=0)
                mean_targets = torch.mean(torch.stack(targets), dim=0)
                model_input.append(mean_inputs.clone().detach())
                model_target.append(mean_targets.clone().detach())

                KL_criterion = nn.KLDivLoss(reduction='sum')
                kl_loss = 0
                for j in range(len(inputs)):
                    kl_loss += KL_criterion(inputs[j], mean_targets) / data_num

                kl_loss = self.beta * kl_loss / len(inputs)
                loss += kl_loss
                loss_list.append(loss)

            for i in range(len(self.models)):
                out_kl_loss = 0
                for j in range(len(model_input)):
                    if(i == j):
                        continue;
                    out_kl_loss += KL_criterion(model_input[i], model_target[j]) / data_num
                out_kl_loss = -0.05 * self.beta * out_kl_loss / (len(model_target) - 1)
                loss_list[i] += out_kl_loss
                losses[i] += loss_list[i].item()
                self.optimizers[i].zero_grad()
                loss_list[i].backward()
                self.optimizers[i].step()

        for i in range(len(self.models)):
            self.schedulers[i].step()


        print_message = 'Epoch [%3d] Total | ' % epoch
        for i in range(len(self.models)):
            print_message += 'Model{i:d}: {loss:.4f}  '.format(
                i=i + 1, loss=losses[i] / (batch_idx + 1))
        tqdm.write(print_message)

        loss_dict = {}
        for i in range(len(self.models)):
            loss_dict[str(i)] = losses[i] / len(self.trainloader)
        self.writer.add_scalars('train/loss', loss_dict, epoch)

    def test(self, epoch):
        for m in self.models:
            m.eval()

        ensemble = Ensemble(self.models)

        loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for _, (inputs, targets, idx) in enumerate(self.testloader):
                inputs, targets = inputs.cuda(), targets.cuda()

                outputs = ensemble(inputs)
                loss += self.criterion(outputs, targets).item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()

                total += inputs.size(0)

        self.writer.add_scalar('test/ensemble_loss', loss / len(self.testloader), epoch)
        self.writer.add_scalar('test/ensemble_acc', 100 * correct / total, epoch)

        print_message = 'Evaluation  | Ensemble Loss {loss:.4f} Acc {acc:.2%}'.format(
            loss=loss / len(self.testloader), acc=correct / total)
        tqdm.write(print_message)

    def save(self, epoch):
        state_dict = {}
        for i, m in enumerate(self.models):
            state_dict['model_%d' % i] = m.state_dict()

        torch.save(state_dict, os.path.join(self.save_root, 'epoch_%d.pth' % epoch))
        tqdm.write(self.save_root)


def get_args():
    parser = argparse.ArgumentParser(description='CIFAR10 FAST Training of Ensemble', add_help=True)
    arguments.model_args(parser)
    arguments.data_args(parser)
    arguments.base_train_args(parser)
    arguments.fast_train_args(parser)
    args = parser.parse_args()
    return args


def main():
    # get args
    args = get_args()

    # set up gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    assert torch.cuda.is_available()

    # set up writer, logger, and save directory for models
    base_dir = 'fasten' if args.num_class == 10 else 'fasten_cifar100'
    save_root = os.path.join('checkpoints', base_dir, 'seed_{:d}'.format(args.seed),
                             'out0.05_{:d}_{:s}{:d}'.format(args.model_num, args.arch, args.depth),
                             'eps{:.2f}_steps{:d}_beta{:.1f}'.format(
                                 args.distill_eps, args.distill_steps, args.beta))
    if args.distill_fixed_layer:
        save_root += '_fixed_layer_{:d}'.format(args.distill_layer)
    if args.plus_adv:
        save_root += '_adv_coeff_{:.1f}_adveps{:.3f}'.format(args.adv_coeff, args.eps)
    if args.start_from == 'scratch':
        save_root += '_start_from_scratch'

    if not os.path.exists(save_root):
        os.makedirs(save_root)
    else:
        print('*********************************')
        print('* The checkpoint already exists *')
        print('*********************************')

    writer = SummaryWriter(save_root.replace('checkpoints', 'runs'))

    # dump configurations for potential future references
    with open(os.path.join(save_root, 'cfg.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=4, sort_keys=True)
    with open(os.path.join(save_root.replace('checkpoints', 'runs'), 'cfg.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=4, sort_keys=True)

    # set up random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # initialize models
    if args.start_from == 'baseline':
        base_dir = 'baseline' if args.num_class == 10 else 'baseline_cifar100'
        args.model_file = os.path.join('checkpoints', base_dir, 'seed_0',
                                       '{:d}_{:s}{:d}'.format(args.model_num, args.arch, args.depth), 'epoch_200.pth')
    elif args.start_from == 'scratch':
        args.model_file = None
    models = my_utils.get_models(args, train=True, as_ensemble=False, model_file=args.model_file, )

    # get data loaders
    trainloader, testloader = my_utils.get_loaders(args)

    # get optimizers and schedulers
    optimizers = my_utils.get_optimizers(args, models)
    schedulers = my_utils.get_schedulers(args, optimizers)

    # train the ensemble
    trainer = FAST_Trainer(models, optimizers, schedulers, trainloader, testloader, writer, save_root, **vars(args))
    trainer.run()


if __name__ == '__main__':
    main()