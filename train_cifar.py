import argparse
import logging
import sys
import time
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from utils import *
from lookahead import *

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

def normalize(X):
    return (X - mu)/std

upper_limit, lower_limit = 1,0

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

class PinballLoss(torch.nn.Module):
    """
    Quantile regression loss
    """
    def __init__(self):
        super(PinballLoss, self).__init__()

    def forward(self, yhat, y, tau=0.5):
        diff = yhat - y
        mask = diff.ge(0).float() - tau
        return (mask * diff).mean()

class Batches():
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last
        )

    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices()
        return ({'input': x.cuda().float(), 'target': y.cuda().long()} for (x,y) in self.dataloader)

    def __len__(self):
        return len(self.dataloader)


def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm, early_stop=False,
               mixup=False, y_a=None, y_b=None, lam=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(normalize(X + delta))
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            if mixup:
                criterion = torch.nn.CrossEntropyLoss()
                loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
            else:
                loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        if mixup:
            criterion = torch.nn.CrossEntropyLoss(reduction='none')
            all_loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
        else:
            all_loss = F.cross_entropy(model(normalize(X+delta)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def get_args():
    parser = argparse.ArgumentParser()
    # Neural Network Config
    parser.add_argument('--arch', default='PreActResNet18')
    parser.add_argument('--arch_version', type=str, default='standard')
    parser.add_argument('--use_FNandWN', action='store_true') # whether use FN and WN
    parser.add_argument('--s_FN', default=15, type=float) # s in FN
    parser.add_argument('--use_FNonly', action='store_true') # whether use FN only
    parser.add_argument('--activation', default='Softplus', type=str)
    parser.add_argument('--softplus_beta', default=10., type=float)
    parser.add_argument('--PYRM_depth', default=32, type=int, help='depth of the network (default: 32)')
    parser.add_argument('--PYRM_alpha', default=300, type=int, help='number of new channel increases per depth (default: 300)')
    parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false', help='to use basicblock for CIFAR datasets (default: bottleneck)')
    parser.set_defaults(bottleneck=True)
    
    # Optimization Config
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--chkpt-iters', default=10, type=int)

    parser.add_argument('--lr-schedule', default='piecewise', choices=['superconverge', 'piecewise', 'linear', 'piecewisesmoothed', 'piecewisezoom', 'onedrop', 'multipledecay', 'cosine'])
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--lr-one-drop', default=0.01, type=float)
    parser.add_argument('--lr-drop-epoch', default=100, type=int)    

    parser.add_argument('--optimizer', default='sgd-m', type=str, choices=['adam', 'sgd-m'])
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='beta1 for adam')
    parser.add_argument('--beta2', default=0.999, type=float,
                        help='beta2 for adam')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup steps for adam')

    # Adversary Config
    parser.add_argument('--test-attack', default='pgd', type=str, choices=['pgd', 'fgsm', 'free','none'])
    parser.add_argument('--train-attack', default='pgd', type=str, choices=['pgd', 'fgsm', 'free', 'none'])
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--attack-iters', default=10, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--pgd-alpha', default=2, type=float)
    parser.add_argument('--fgsm-alpha', default=1.25, type=float)
    parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--fgsm-init', default='random', choices=['zero', 'random', 'previous'])
    parser.add_argument('--fname', default='./', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--width-factor', default=10, type=int)
    parser.add_argument('--resume', default=0, type=int)

    # Data Augmentation Config
    parser.add_argument('--cutout', action='store_true', default=False)
    parser.add_argument('--cutout-len', type=int, default=14)
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--mixup-alpha', type=float)
    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--l1', default=0, type=float)

    # Flooding Config
    parser.add_argument('--flooding', action='store_true', default=False)
    parser.add_argument('--rand_b', action='store_true')
    parser.add_argument('--b', default=1.2, type=float)
    parser.add_argument('--b1', default=0.8, type=float)
    parser.add_argument('--b2', default=1.2, type=float)
    parser.add_argument('--beta', default=6., type=float)
    parser.add_argument('--beta_ada', default=1., type=float)
    parser.add_argument('--fl_best', action='store_true')
    
    # Lookahead Config
    parser.add_argument('--lookahead', action='store_true', default=False)
    parser.add_argument('--la_steps', default=8, type=float, help='warmup steps for adam')
    parser.add_argument('--la_alpha', default=0.5, type=float, help='warmup steps for adam')

    # Uncertainty Config
    parser.add_argument('--Pinball', action='store_true', default=True) 
    parser.add_argument('--tau', default=0.1, type=float, help='tau for Pinball') 
    parser.add_argument('--ada_tau', action='store_true') 
    parser.add_argument('--tau_1', default=0.7, type=float)
    parser.add_argument('--tau_2', default=0.9, type=float)
    parser.add_argument('--beta_pinball', default=6., type=float, help='tau for Pinball') 

    return parser.parse_args()

global_step = 0

def main():
    args = get_args()

    if not os.path.exists(args.fname):
        os.makedirs(args.fname)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.fname, 'eval.log' if args.eval else 'output.log')),
            logging.StreamHandler()
        ])

    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    transforms = [Crop(32, 32), FlipLR()]
    if args.cutout:
        transforms.append(Cutout(args.cutout_len, args.cutout_len))
    if args.val:
        try:
            dataset = torch.load("cifar10_validation_split.pth")
        except:
            print("Couldn't find a dataset with a validation split, did you run "
                  "generate_validation.py?")
            return
        val_set = list(zip(transpose(dataset['val']['data']/255.), dataset['val']['labels']))
        val_batches = Batches(val_set, args.batch_size, shuffle=False, num_workers=2)
    else:
        dataset = cifar10(args.data_dir)
        
    train_set = list(zip(transpose(pad(dataset['train']['data'], 4)/255.),
        dataset['train']['labels']))
    train_set_x = Transform(train_set, transforms)
    train_batches = Batches(train_set_x, args.batch_size, shuffle=True, set_random_choices=True, num_workers=2)

    test_set = list(zip(transpose(dataset['test']['data']/255.), dataset['test']['labels']))
    test_batches = Batches(test_set, args.batch_size, shuffle=False, num_workers=2)

    epsilon = (args.epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)

    num_classes = 10
    if args.arch_version == 'smooth':
        from wideresnet import WideResNet
        from preactresnet import PreActResNet18, PreActResNet50
        if args.arch == 'WideResNet34':
            # net = WideResNet(depth=34, num_classes=num_classes, widen_factor=10)
            model = WideResNet(depth=34, num_classes=num_classes, widen_factor=20, dropRate=0.0, normalize=args.use_FNandWN, activation=args.activation, softplus_beta=args.softplus_beta)
            # ema_model = WideResNet(depth=34, num_classes=num_classes, widen_factor=20, dropRate=0.0, normalize=args.use_FNandWN, activation=args.activation, softplus_beta=args.softplus_beta)
            # model = WideResNet(34, 10, widen_factor=10, dropRate=0.0, normalize=args.use_FNandWN, activation=args.activation, softplus_beta=args.softplus_beta)
        elif args.arch == 'WideResNet28':
            model = WideResNet(depth=28, num_classes=num_classes, widen_factor=20, dropRate=0.0, normalize=args.use_FNandWN, activation=args.activation, softplus_beta=args.softplus_beta)
            # ema_model = WideResNet(depth=28, num_classes=num_classes, widen_factor=20, dropRate=0.0, normalize=args.use_FNandWN, activation=args.activation, softplus_beta=args.softplus_beta)
        elif args.arch == 'PreActResNet18':
            # net = PreActResNet18(num_classes=num_classes)
            model = PreActResNet18(normalize_only_FN=args.use_FNonly, normalize=args.use_FNandWN, scale=args.s_FN, activation=args.activation, softplus_beta=args.softplus_beta)
            # ema_model = PreActResNet18(normalize_only_FN=args.use_FNonly, normalize=args.use_FNandWN, scale=args.s_FN, activation=args.activation, softplus_beta=args.softplus_beta)
        elif args.arch == 'ResNet50':
            model = PreActResNet50(normalize_only_FN=args.use_FNonly, normalize=args.use_FNandWN, scale=args.s_FN, activation=args.activation, softplus_beta=args.softplus_beta)
    elif args.arch_version == 'standard':
        from wideresnet_1 import WideResNet
        from preactresnet_1 import PreActResNet18, PreActResNet50
        num_classes = 10
        if args.arch == 'WideResNet34':
            model = WideResNet(depth=34, num_classes=num_classes, widen_factor=10, dropRate=0.2)
            # ema_model = WideResNet(depth=34, num_classes=num_classes, widen_factor=20)
        elif args.arch == 'WideResNet28':
            model = WideResNet(depth=28, num_classes=num_classes, widen_factor=20, dropRate=0.2)
            # ema_model = WideResNet(depth=28, num_classes=num_classes, widen_factor=10)
        elif args.arch == 'PreActResNet18':
            model = PreActResNet18()
        elif args.arch == 'ResNet50':
            model = PreActResNet50()
            # ema_model = PreActResNet18()
    else:
        raise ValueError('Please use choose correct architectures.')

    model = torch.nn.DataParallel(model)
    model.train()

    if args.l2:
        decay, no_decay = [], []
        for name,param in model.named_parameters():
            if 'bn' not in name and 'bias' not in name:
                decay.append(param)
            else:
                no_decay.append(param)
        params = [{'params':decay, 'weight_decay':args.l2},
                  {'params':no_decay, 'weight_decay': 0 }]
    else:
        params = model.parameters()

    if args.optimizer.lower() == 'sgd-m':
        opt = torch.optim.SGD(params, lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adam':
        opt = torch.optim.Adam(params, lr=args.lr_max, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    
    if args.lookahead:
        opt = Lookahead(opt, la_steps=args.la_steps, la_alpha=args.la_alpha)
        print('LookaHead')

    criterion = torch.nn.CrossEntropyLoss()
    pinball_criterion = PinballLoss()
    criterion_kl = nn.KLDivLoss(size_average=False)

    if args.train_attack == 'free':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
        delta.requires_grad = True
    elif args.train_attack == 'fgsm' and args.fgsm_init == 'previous':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
        delta.requires_grad = True

    if args.train_attack == 'free':
        epochs = int(math.ceil(args.epochs / args.attack_iters))
    else:
        epochs = args.epochs

    if args.lr_schedule == 'superconverge':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr_max, 0])[0]
    elif args.lr_schedule == 'piecewise':
        def lr_schedule(t):
            if t / args.epochs < 0.5:
                return args.lr_max
            elif t / args.epochs < 0.75:
                return args.lr_max / 10.
            else:
                return args.lr_max / 100.
    elif args.lr_schedule == 'linear':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs // 3, args.epochs * 2 // 3, args.epochs], [args.lr_max, args.lr_max, args.lr_max / 10, args.lr_max / 100])[0]
    elif args.lr_schedule == 'onedrop':
        def lr_schedule(t):
            if t < args.lr_drop_epoch:
                return args.lr_max
            else:
                return args.lr_one_drop
    elif args.lr_schedule == 'multipledecay':
        def lr_schedule(t):
            return args.lr_max - (t//(args.epochs//10))*(args.lr_max/10)
    elif args.lr_schedule == 'cosine': 
        def lr_schedule(t): 
            return args.lr_max * 0.5 * (1 + np.cos(t / args.epochs * np.pi))


    best_test_robust_acc = 0
    best_val_robust_acc = 0
    if args.resume:
        start_epoch = args.resume
        model.load_state_dict(torch.load(os.path.join(args.fname, f'model_{start_epoch-1}.pth')))
        opt.load_state_dict(torch.load(os.path.join(args.fname, f'opt_{start_epoch-1}.pth')))
        logger.info(f'Resuming at epoch {start_epoch}')

        best_test_robust_acc = torch.load(os.path.join(args.fname, f'model_best.pth'))['test_robust_acc']
        if args.val:
            best_val_robust_acc = torch.load(os.path.join(args.fname, f'model_val.pth'))['val_robust_acc']
    else:
        start_epoch = 0

    if args.eval:
        if not args.resume:
            logger.info("No model loaded to evaluate, specify with --resume FNAME")
            return
        logger.info("[Evaluation mode]")

    if args.Pinball:
        logger.info('Epoch \t Train Time \t Test Time \t LR \t \t Train Loss \t Train Acc \t Train Robust Loss \t Train Robust Acc \t Test Loss \t Test Acc \t Test Robust Loss \t Test Robust Acc \t AU1 \t AU2 \t Train_Pinball_loss')
    else:
        logger.info('Epoch \t Train Time \t Test Time \t LR \t \t Train Loss \t Train Acc \t Train Robust Loss \t Train Robust Acc \t Test Loss \t Test Acc \t Test Robust Loss \t Test Robust Acc \t R')
        
    for epoch in range(start_epoch, epochs):
        model.train()
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_robust_loss = 0
        train_robust_acc = 0
        train_pinball_loss = 0
        train_au1 = 0
        train_au2 = 0
        train_n = 0

        fl_b_tmp = 2.

        for i, batch in enumerate(train_batches):
            if args.eval:
                break
            X, y = batch['input'], batch['target']
            if args.mixup:
                X, y_a, y_b, lam = mixup_data(X, y, args.mixup_alpha)
                X, y_a, y_b = map(Variable, (X, y_a, y_b))
            lr = lr_schedule(epoch + (i + 1) / len(train_batches))
            opt.param_groups[0].update(lr=lr)

            if args.train_attack == 'pgd':
                if args.mixup:
                    delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm, mixup=True, y_a=y_a, y_b=y_b, lam=lam)
                else:
                    delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm)
                delta = delta.detach()
            elif args.train_attack == 'fgsm':
                delta = attack_pgd(model, X, y, epsilon, args.fgsm_alpha*epsilon, 1, 1, args.norm)
            elif args.train_attack == 'none':
                delta = torch.zeros_like(X)

            robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
            natural_output = model(normalize(X))

            if args.mixup:
                criterion = nn.CrossEntropyLoss(reduction='mean')()
                robust_loss = mixup_criterion(criterion, natural_output, y_a, y_b, lam)
            else:
                robust_loss = nn.CrossEntropyLoss(reduction='mean')(natural_output, y)

            if args.l1:
                for name,param in model.named_parameters():
                    if 'bn' not in name and 'bias' not in name:
                        robust_loss += args.l1*param.abs().sum()
            
            if args.flooding:
                if args.rand_b:
                    randB = np.random.uniform(args.b1, args.b2)
                    robust_loss = (robust_loss-randB).abs()+randB
                elif args.fl_best and args.val:
                    robust_loss = (robust_loss-fl_b_tmp).abs()+fl_b_tmp
                else:
                    robust_loss = (robust_loss-args.b).abs()+args.b
            
            if args.Pinball:
                AU1 = nn.NLLLoss(reduction='none')(robust_output, y)
                AU2 = nn.NLLLoss(reduction='none')(natural_output, y)
                if args.ada_tau:
                    tau_tmp = np.random.uniform(args.tau_1, args.tau_2)
                    pinball_loss = args.beta_pinball * pinball_criterion(natural_output, robust_output, tau_tmp)
                    robust_loss += pinball_loss
                else:
                    pinball_loss = args.beta_pinball * pinball_criterion(natural_output, robust_output, args.tau)
                    # pinball_loss = args.beta_pinball * criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_natural, dim=1))
                    # pinball_loss = args.beta_pinball * (1.0 / args.batch_size) * criterion_kl(F.log_softmax(robust_output, dim=1), F.softmax(natural_output, dim=1))
                    robust_loss += pinball_loss

            opt.zero_grad()
            robust_loss.backward()
            opt.step()
            output = model(normalize(X))

            if args.mixup:
                loss = mixup_criterion(criterion, output, y_a, y_b, lam)
            else:
                loss = criterion(output, y)

            train_robust_loss += robust_loss.item() * y.size(0)
            train_robust_acc += (robust_output.max(1)[1] == y).sum().item()
            train_loss += loss.item() * y.size(0) 
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            if args.Pinball:
                train_pinball_loss += pinball_loss.item() * y.size(0)
                train_au1 += AU1.mean().item() * y.size(0)
                train_au2 += AU2.mean().item() * y.size(0)

        train_time = time.time()

        model.eval()
        test_loss = 0
        test_acc = 0
        test_robust_loss = 0
        test_robust_acc = 0
        test_n = 0
        test_pgd_alpha = 2./255.
        for i, batch in enumerate(test_batches):
            X, y = batch['input'], batch['target']

            if args.test_attack == 'none':
                delta = torch.zeros_like(X)
            else:
                delta = attack_pgd(model, X, y, epsilon, pgd_alpha, 10, args.restarts, args.norm, early_stop=args.eval)

            delta = delta.detach()
            with torch.no_grad():
                robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
                robust_loss = criterion(robust_output, y)

                output = model(normalize(X))
                loss = criterion(output, y)

            test_robust_loss += robust_loss.item() * y.size(0)
            test_robust_acc += (robust_output.max(1)[1] == y).sum().item()
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            test_n += y.size(0)

        test_time = time.time()

        if args.val:
            val_loss = 0
            val_acc = 0
            val_robust_loss = 0
            val_robust_acc = 0
            val_n = 0
            for i, batch in enumerate(val_batches):
                X, y = batch['input'], batch['target']

                if args.test_attack == 'none':
                    delta = torch.zeros_like(X)
                else:
                    delta = attack_pgd(model, X, y, epsilon, pgd_alpha, 10, args.restarts, args.norm, early_stop=args.eval)
                delta = delta.detach()

                robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
                robust_loss = criterion(robust_output, y)

                output = model(normalize(X))
                loss = criterion(output, y)

                val_robust_loss += robust_loss.item() * y.size(0)
                val_robust_acc += (robust_output.max(1)[1] == y).sum().item()
                val_loss += loss.item() * y.size(0)
                val_acc += (output.max(1)[1] == y).sum().item()
                val_n += y.size(0)
                
                if fl_b_tmp > val_robust_loss:
                    fl_b_tmp = val_robust_loss / val_n


        if not args.eval:
            if args.Pinball:
                logger.info('%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
                epoch, train_time - start_time, test_time - train_time, lr,
                train_loss/train_n, train_acc/train_n, train_robust_loss/train_n, train_robust_acc/train_n,
                test_loss/test_n, test_acc/test_n, test_robust_loss/test_n, test_robust_acc/test_n, train_au1/train_n, train_au2/train_n, train_pinball_loss/train_n)
            else:
                logger.info('%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f \t \t %.4f \t %.4f \t %.4f \t \t %.4f',
                    epoch, train_time - start_time, test_time - train_time, lr,
                    train_loss/train_n, train_acc/train_n, train_robust_loss/train_n, train_robust_acc/train_n,
                    test_loss/test_n, test_acc/test_n, test_robust_loss/test_n, test_robust_acc/test_n)

            if args.val:
                logger.info('validation %.4f \t %.4f \t %.4f \t %.4f',
                    val_loss/val_n, val_acc/val_n, val_robust_loss/val_n, val_robust_acc/val_n)

                if val_robust_acc/val_n > best_val_robust_acc:
                    torch.save({
                            'state_dict':model.state_dict(),
                            'test_robust_acc':test_robust_acc/test_n,
                            'test_robust_loss':test_robust_loss/test_n,
                            'test_loss':test_loss/test_n,
                            'test_acc':test_acc/test_n,
                            'val_robust_acc':val_robust_acc/val_n,
                            'val_robust_loss':val_robust_loss/val_n,
                            'val_loss':val_loss/val_n,
                            'val_acc':val_acc/val_n,
                        }, os.path.join(args.fname, f'model_val.pth'))
                    best_val_robust_acc = val_robust_acc/val_n

            # save checkpoint
            if (epoch+1) % args.chkpt_iters == 0 or epoch+1 == epochs:
                torch.save(model.state_dict(), os.path.join(args.fname, f'model_{epoch}.pth'))
                torch.save(opt.state_dict(), os.path.join(args.fname, f'opt_{epoch}.pth'))

            # save best
            if test_robust_acc/test_n > best_test_robust_acc:
                torch.save({
                        'state_dict':model.state_dict(),
                        'test_robust_acc':test_robust_acc/test_n,
                        'test_robust_loss':test_robust_loss/test_n,
                        'test_loss':test_loss/test_n,
                        'test_acc':test_acc/test_n,
                    }, os.path.join(args.fname, f'model_best.pth'))
                best_test_robust_acc = test_robust_acc/test_n
        else:
            logger.info('%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f \t \t %.4f \t %.4f \t %.4f \t \t %.4f',
                epoch, train_time - start_time, test_time - train_time, -1,
                -1, -1, -1, -1,
                test_loss/test_n, test_acc/test_n, test_robust_loss/test_n, test_robust_acc/test_n)
            return


if __name__ == "__main__":
    main()
