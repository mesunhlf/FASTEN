import os, sys

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
sys.path.append(os.getcwd())
import json, argparse, random, time
from tqdm import tqdm
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import arguments
import  utils
from models.ensemble import Ensemble
from distillation import Linf_PGD, Linf_distillation
from models.resnet import ResNet


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def requires_grad_(model: torch.nn.Module, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)


def Cosine(g1, g2):
    return torch.abs(F.cosine_similarity(g1, g2)).mean()  # + (0.05 * torch.sum(g1**2+g2**2,1)).mean()


def Magnitude(g1):
    return (torch.sum(g1 ** 2, 1)).mean() * 2


def PGD(models, inputs, labels, eps):
    # steps = 6
    # alpha = eps / 3.
    steps = 3
    alpha = (2*eps) / 3.

    adv = inputs.detach() + torch.FloatTensor(inputs.shape).uniform_(-eps, eps).cuda()
    adv = torch.clamp(adv, 0, 1)
    criterion = nn.CrossEntropyLoss()

    adv.requires_grad = True
    for _ in range(steps):
        # adv.requires_grad_()
        grad_loss = 0
        for i, m in enumerate(models):
            loss = criterion(m(adv), labels)
            grad = autograd.grad(loss, adv, create_graph=True)[0]
            grad = grad.flatten(start_dim=1)
            grad_loss += Magnitude(grad)

        grad_loss /= len(models)
        grad_loss.backward()
        sign_grad = adv.grad.data.sign()
        with torch.no_grad():
            adv.data = adv.data + alpha * sign_grad
            adv.data = torch.max(torch.min(adv.data, inputs + eps), inputs - eps)
            adv.data = torch.clamp(adv.data, 0., 1.)

    adv.grad = None
    return adv.detach()


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# class NormalizeLayer(torch.nn.Module):
#     """Standardize the channels of a batch of images by subtracting the dataset mean
#         and dividing by the dataset standard deviation.
#
#         In order to certify radii in original coordinates rather than standardized coordinates, we
#         add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
#         layer of the classifier rather than as a part of preprocessing as is typical."""
#
#     def __init__(self, means, sds):
#         """
#         :param means: the channel means
#         :param sds: the channel standard deviations
#         """
#         super(NormalizeLayer, self).__init__()
#         self.means = torch.tensor(means).cuda()
#         self.sds = torch.tensor(sds).cuda()
#
#     def forward(self, input: torch.tensor):
#         (batch_size, num_channels, height, width) = input.shape
#         means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2).cuda()
#         sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2).cuda()
#         # print(input)
#         return (input - means) / sds
#
#
# def get_normalize_layer() -> torch.nn.Module:
#     """Return the dataset's normalization layer"""
#     _CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
#     _CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]
#     return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)
#
#
# def get_architecture() -> torch.nn.Module:
#     """ Return a neural network (with random weights)
#
#     :param arch: the architecture - should be in the ARCHITECTURES list above
#     :param dataset: the dataset - should be in the datasets.DATASETS list
#     :return: a Pytorch module
#     """
#     model = ResNet(depth=20, num_classes=10)
#     normalize_layer = get_normalize_layer()
#     return torch.nn.Sequential(normalize_layer, model).cuda()


class TRS_Trainer():
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
        # self.distill_fixed_layer = kwargs['distill_fixed_layer']
        # self.distill_cfg = {'eps': kwargs['distill_eps'],
        #                     'alpha': kwargs['distill_alpha'],
        #                     'steps': kwargs['distill_steps'],
        #                     'layer': kwargs['distill_layer'],
        #                     'rand_start': kwargs['distill_rand_start'],
        #                     'before_relu': True,
        #                     'momentum': kwargs['distill_momentum']
        #                     }

        # diversity training configs
        self.plus_adv = kwargs['plus_adv']
        self.coeff = kwargs['coeff']
        self.adv_eps = kwargs['adv_eps']
        self.init_eps = kwargs['init_eps']
        self.lamda = kwargs['lamda']
        self.scale = kwargs['scale']
        if self.plus_adv:
            self.attack_cfg = {'eps': kwargs['eps'],
                               'alpha': kwargs['alpha'],
                               'steps': kwargs['steps'],
                               'is_targeted': False,
                               'rand_start': True
                               }
        self.depth = kwargs['depth']

    def get_epoch_iterator(self):
        iterator = tqdm(list(range(1, self.epochs + 1)), total=self.epochs, desc='Epoch',
                        leave=True, position=1)
        return iterator

    def get_batch_iterator(self):
        iterator = tqdm(self.trainloader, desc='Batch', leave=False, position=2)
        return iterator

    def run(self):
        epoch_iter = self.get_epoch_iterator()
        for epoch in epoch_iter:
            if (epoch==1):
                self.test(epoch)
            self.train(epoch)
            self.test(epoch)
            self.schedulers.step(epoch)
            self.save(epoch)

    def train(self, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        cos_losses = AverageMeter()
        smooth_losses = AverageMeter()
        cos01_losses = AverageMeter()
        cos02_losses = AverageMeter()
        cos12_losses = AverageMeter()

        end = time.time()

        for m in self.models:
            m.train()
            requires_grad_(m, True)

        batch_iter = self.get_batch_iterator()
        for i, (inputs, targets) in enumerate(batch_iter):
            # measure data loading time

            data_time.update(time.time() - end)

            inputs, targets = inputs.cuda(), targets.cuda()
            batch_size = inputs.size(0)
            inputs.requires_grad = True
            grads = []
            loss_std = 0
            for j in range(len(self.models)):
                logits = self.models[j](inputs)
                loss = self.criterion(logits, targets)
                grad = autograd.grad(loss, inputs, create_graph=True)[0]
                grad = grad.flatten(start_dim=1)
                grads.append(grad)
                loss_std += loss

            cos_loss, smooth_loss = 0, 0

            cos01 = Cosine(grads[0], grads[1])
            cos02 = Cosine(grads[0], grads[2])
            # cos03 = Cosine(grads[0], grads[3])
            # cos04 = Cosine(grads[0], grads[4])
            cos12 = Cosine(grads[1], grads[2])
            # cos13 = Cosine(grads[1], grads[3])
            # cos14 = Cosine(grads[1], grads[4])
            # cos23 = Cosine(grads[2], grads[3])
            # cos24 = Cosine(grads[2], grads[4])
            # cos34 = Cosine(grads[3], grads[4])
            #
            # cos05 = Cosine(grads[0], grads[5])
            # cos06 = Cosine(grads[0], grads[6])
            # cos07 = Cosine(grads[0], grads[7])
            # cos15 = Cosine(grads[1], grads[5])
            # cos16 = Cosine(grads[1], grads[6])
            # cos17 = Cosine(grads[1], grads[7])
            # cos25 = Cosine(grads[2], grads[5])
            # cos26 = Cosine(grads[2], grads[6])
            # cos27 = Cosine(grads[2], grads[7])
            # cos35 = Cosine(grads[3], grads[5])
            # cos36 = Cosine(grads[3], grads[6])
            # cos37 = Cosine(grads[3], grads[7])
            # cos45 = Cosine(grads[4], grads[5])
            # cos46 = Cosine(grads[4], grads[6])
            # cos47 = Cosine(grads[4], grads[7])
            # cos56 = Cosine(grads[5], grads[6])
            # cos57 = Cosine(grads[5], grads[7])
            # cos67 = Cosine(grads[6], grads[7])

            cos_loss = (cos01 + cos02 + cos12) / 3.
            # cos_loss = (cos01 + cos02 + cos03 + cos04 + cos12 + cos13 + cos14 + cos23 + cos24 + cos34) / 10.
            # cos_loss = (cos01 + cos02 + cos03 + cos04 + cos12 + cos13 + cos14 + cos23 + cos24 + cos34 + cos05
            #             + cos06 + cos07 + cos15 + cos16 + cos17 + cos25 + cos26 + cos27 + cos35 + cos36 + cos37
            #             + cos45 + cos46 + cos47 + cos56 + cos57 + cos67) / 28.

            N = inputs.shape[0] // 2
            cureps = (self.adv_eps - self.init_eps) * epoch / self.epochs + self.init_eps
            clean_inputs = inputs[:N].detach()  # PGD(self.models, inputs[:N], targets[:N])
            adv_inputs = PGD(self.models, inputs[N:], targets[N:], cureps).detach()

            adv_x = torch.cat([clean_inputs, adv_inputs])

            adv_x.requires_grad = True

            # loss_advt = 0
            if self.plus_adv:
                for j in range(len(self.models)):
                    outputs = self.models[j](adv_x)
                    loss = self.criterion(outputs, targets)
                    grad = autograd.grad(loss, adv_x, create_graph=True)[0]
                    grad = grad.flatten(start_dim=1)
                    smooth_loss += Magnitude(grad)
                    # loss_advt += loss
                    # loss_advt += self.criterion(self.models[j](PGD(self.models, inputs, targets, 0.07).detach()), targets)

            else:
                # grads = []
                for j in range(len(self.models)):
                    outputs = self.models[j](inputs)
                    loss = self.criterion(outputs, targets)
                    grad = autograd.grad(loss, inputs, create_graph=True)[0]
                    grad = grad.flatten(start_dim=1)
                    smooth_loss += Magnitude(grad)

            smooth_loss /= len(self.models)


            loss = loss_std + self.scale * (self.coeff * cos_loss + self.lamda * smooth_loss) # + (loss_advt / len(self.models))
            #coeff:20,lambda:2.5,scale:1.0

            ensemble = Ensemble(self.models)
            logits = ensemble(inputs)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
            losses.update(loss.item(), batch_size)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)
            cos_losses.update(cos_loss.item(), batch_size)
            smooth_losses.update(smooth_loss.item(), batch_size)
            cos01_losses.update(cos01.item(), batch_size)
            cos02_losses.update(cos02.item(), batch_size)
            cos12_losses.update(cos12.item(), batch_size)
            # print(i)
            if (i % 200 == 0):
                print("total_loss:", losses.avg)
                print("smooth_loss:", smooth_losses.avg)
                print("cos_loss:", cos_losses.avg)

            self.optimizers.zero_grad()
            loss.backward()
            self.optimizers.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            #     print('Epoch: [{0}][{1}/{2}]\t'
            #           'Time {batch_time.avg:.3f}\t'
            #           'Data {data_time.avg:.3f}\t'
            #           'Loss {loss.avg:.4f}\t'
            #           'Acc@1 {top1.avg:.3f}\t'
            #           'Acc@5 {top5.avg:.3f}'.format(
            #         epoch, i, len(loader), batch_time=batch_time,
            #         data_time=data_time, loss=losses, top1=top1, top5=top5))

        print_message = 'Epoch [%3d] | ' % epoch
        print_message += 'Loss {loss.avg:.4f}   Acc@1 {top1.avg:.4f}    Acc@5 {top5.avg:.4f}'.format(
            loss=losses, top1=top1, top5=top5)
        tqdm.write(print_message)

        self.writer.add_scalar('train/batch_time', batch_time.avg, epoch)
        self.writer.add_scalar('train/acc@1', top1.avg, epoch)
        self.writer.add_scalar('train/acc@5', top5.avg, epoch)
        self.writer.add_scalar('train/loss', losses.avg, epoch)
        self.writer.add_scalar('train/cos_loss', cos_losses.avg, epoch)
        self.writer.add_scalar('train/smooth_loss', smooth_losses.avg, epoch)
        self.writer.add_scalar('train/cos01', cos01_losses.avg, epoch)
        self.writer.add_scalar('train/cos02', cos02_losses.avg, epoch)
        self.writer.add_scalar('train/cos12', cos12_losses.avg, epoch)

    def test(self, epoch):
        for m in self.models:
            m.eval()

        ensemble = Ensemble(self.models)

        loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for _, (inputs, targets) in enumerate(self.testloader):
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


def get_args():
    parser = argparse.ArgumentParser(description='CIFAR10 DVERGE Training of Ensemble', add_help=True)
    arguments.model_args(parser)
    arguments.data_args(parser)
    arguments.base_train_args(parser)
    arguments.trs_train_args(parser)
    args = parser.parse_args()
    return args


def main():
    # get args
    args = get_args()
    print(args)

    # set up gpus
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # assert torch.cuda.is_available()
    torch.cuda.set_device(0)

    # set up writer, logger, and save directory for models
    # base_dir = 'trs' if args.num_class == 10 else 'trs_cifar100'
    save_root = os.path.join('checkpoints', 'trs', 'seed_{:d}'.format(args.seed),
                             '{:d}_{:s}{:d}_eps_{:.2f}_{:.2f}_adam'.format(
                                 args.model_num, args.arch, args.depth, args.init_eps, args.adv_eps))
    if args.plus_adv:
        save_root += '_plus_adv_coeff_{:.1f}_4'.format(args.coeff)
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
    # if args.start_from == 'baseline':
    #     base_dir = 'baseline' if args.num_class == 10 else 'baseline_cifar100'
    #     args.model_file = os.path.join('checkpoints', base_dir, 'seed_0',
    #                                    '{:d}_{:s}{:d}'.format(args.model_num, args.arch, args.depth), 'epoch_200.pth')
    # elif args.divtrain_start_from == 'scratch':
    #     args.model_file = None
    args.model_file = "/home/qiupeichao/Accept/DVERGE-main/FASTEN/checkpoints/baseline/seed_0/3_ResNet20/epoch_2.pth"
    models_r = utils.get_models(args, train=True, as_ensemble=False, model_file=args.model_file)

    # get data loaders
    trainloader, testloader = utils.get_loaders(args)

    models = []
    for i in range(args.model_num):
        submodel = nn.DataParallel(models_r[i],device_ids=[0])
        models.append(submodel)
    print("Model loaded")

    # get optimizers and schedulers
    # optimizers = utils.get_optimizers(args, models)
    # schedulers = utils.get_schedulers(args, optimizers)
    param = list(models[0].parameters())
    for i in range(1, args.model_num):
        param.extend(list(models[i].parameters()))
    optimizers = optim.SGD(param, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # optimizers = optim.Adam(param, lr=args.lr, weight_decay=1e-4, eps=1e-7)
    schedulers = optim.lr_scheduler.MultiStepLR(optimizers, milestones=args.sch_intervals, gamma=args.lr_gamma)

    # if args.start_from == 'baseline':
    #     base_dir = 'baseline' if args.num_class == 10 else 'baseline_cifar100'
    #     base_classifier = os.path.join('checkpoints', base_dir, 'seed_0',
    #                                    '{:d}_{:s}{:d}'.format(args.model_num, args.arch, args.depth), 'epoch_200.pth')
    #     state_dict = torch.load(base_classifier)
    #     # iter_m = state_dict.keys()
    #     for i in range(args.model_num):
    #         models[i].load_state_dict(state_dict['state_dict'])
    #         models[i].train()
    #     print("Loaded...")

    # train the ensemble
    trainer = TRS_Trainer(models, optimizers, schedulers, trainloader, testloader, writer, save_root, **vars(args))
    trainer.run()


if __name__ == '__main__':
    main()
