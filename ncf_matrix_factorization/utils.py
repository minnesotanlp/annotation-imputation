import os
import sys
import time
from datetime import datetime
import shutil
import math
import argparse

import numpy as np
import torch
import random
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def get_output_file(args):
    start = args.start_index
    end = args.start_index + args.num_samples

    if args.data_type == 'mnli':
        dataset_str = f"{args.data_type}_{args.mnli_option}_{args.attack_target}"
    else:
        dataset_str = args.data_type
    attack_str = args.adv_loss
    if args.adv_loss == 'cw':
        attack_str += f'_kappa={args.kappa}'

    output_file = f"{args.model_name}_{dataset_str}_{start}-{end}"
    output_file += f"_iters={args.num_iters}_{attack_str}_lambda_sim={args.lam_sim}_lambda_perp={args.lam_perp}" \
                   f"_lambda_pref={args.lam_pref}_{args.constraint}.pth"

    return output_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Logger(object):
    """Reference: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514"""
    def __init__(self, fn):
        if not os.path.exists("./logs/"):
            os.mkdir("./logs/")

        logdir = 'logs/' + fn
        if not os.path.exists(logdir):
            os.makedirs(logdir) # [London] replaced mkdir with makedirs
        if len(os.listdir(logdir)) != 0:
            ans = input(f"log_dir '{logdir}' is not empty. All data inside log_dir will be deleted. "
                            "Will you proceed [y/N]? ")
            if ans in ['y', 'Y']:
                shutil.rmtree(logdir)
            else:
                exit(1)
        self.set_dir(logdir)

    def set_dir(self, logdir, log_fn='log.txt'):
        self.logdir = logdir
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        self.log_file = open(os.path.join(logdir, log_fn), 'a')

    def log(self, string):
        self.log_file.write('[%s] %s' % (datetime.now().isoformat(" ", "seconds"), string) + '\n')
        self.log_file.flush()

        print('[%s] %s' % (datetime.now().isoformat(" ", "seconds"), string))
        sys.stdout.flush()

    def log_dirname(self, string):
        self.log_file.write('%s (%s)' % (string, self.logdir) + '\n')
        self.log_file.flush()

        print('%s (%s)' % (string, self.logdir))
        sys.stdout.flush()

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def set_model_path(args, dataset):
    # Naming the saving model
    suffix = "_"
    suffix += str(args.train_type)

    return args.dataset + suffix + 'model'

def save_model(args, model, log_dir, dataset):
    # Save the model
    if isinstance(model, nn.DataParallel):
        model = model.module

    os.makedirs(log_dir, exist_ok=True)
    #model_path = set_model_path(args, dataset)
    save_path = os.path.join(log_dir, 'model')
    torch.save(model.state_dict(), save_path)

def cut_input(args, tokens):
    if 'roberta' not in args.backbone:
        attention_mask = (tokens != 1).float()
    else:
        attention_mask = (tokens > 0).float()
    max_len = int(torch.max(attention_mask.sum(dim=1)))
    return tokens[:, :max_len]
    
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_annotations", help='location of the train dataset annotations', required=True)
    parser.add_argument("--validation_annotations", help='location of the validation dataset annotations', required=True)
    parser.add_argument("--train_texts", help='location of the train dataset texts', required=True)
    parser.add_argument("--validation_texts", help='location of the validation dataset texts', required=True)
    parser.add_argument("--dataset_name", help='dataset name',
                        required=False, type=str, default='Dataset')
    parser.add_argument("--backbone", help='backbone network',
                        default='roberta-base', type=str)
    parser.add_argument("--seed", help='random seed',
                        default=0, type=int)

    parser.add_argument("--train_type", help='training details',
                        default='base', type=str)
    parser.add_argument("--epochs", help='training epochs',
                        default=20, type=int)
    parser.add_argument("--batch_size", help='training bacth size',
                        default=16, type=int)
    parser.add_argument("--model_lr", help='learning rate for model update',
                        default=1e-5, type=float)
    parser.add_argument("--save_ckpt", help='save the best model checkpoint',
                        action='store_true')
    parser.add_argument("--pre_ckpt", help='path for the pre-trained model',
                        default=None, type=str)

    parser.add_argument("--method", help='baseline methods',
                        choices=['hard', 'soft', 'margin', 'filtering', 'weight', 'cskd', 'multi'],
                        default='hard', type=str)
    parser.add_argument("--n_anno", help='number of annotator',
                        default=1, type=int)
    parser.add_argument("--gen_anno", help='path for the imputed annotations',
                        default=None, type=str)
    parser.add_argument("--advanced_logger_type", help='advanced logger type', default='tensorboard', type=str)
    parser.add_argument("--wandb_project", help='wandb project name', default=None, type=str)
    parser.add_argument("--wandb_run_name", help='wandb run name', default='default_run_name', type=str)

    return parser.parse_args()