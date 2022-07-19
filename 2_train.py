# Copyright (c) 2022, Kwanhyung Lee, Hyewon Jeong, Seyun Kim AITRICS. All rights reserved.
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import argparse
import random
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from itertools import groupby
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
from torchsummary import summary
from torchinfo import summary

from builder.utils.lars import LARC
from control.config import args

from builder.data.data_preprocess import get_data_preprocessed
# from builder.data.data_preprocess_temp1 import get_data_preprocessed
from builder.models import get_detector_model, grad_cam
from builder.utils.logger import Logger
from builder.utils.utils import set_seeds, set_devices
from builder.utils.cosine_annealing_with_warmup import CosineAnnealingWarmUpRestarts
from builder.utils.cosine_annealing_with_warmupSingle import CosineAnnealingWarmUpSingle
from builder.trainer import get_trainer
from builder.trainer import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
list_of_test_results_per_seed = []

val_per_epochs = 10
epoch_losses =[]
loss = 0

for seed_num in args.seed_list:
    args.seed = seed_num
    set_seeds(args)
    device = set_devices(args)
    print(device)
    logger = Logger(args)
    logger.evaluator.best_auc = 0

    # Load Data, Create Model 
    train_loader, val_loader, test_loader, len_train_dir, len_val_dir, len_test_dir = get_data_preprocessed(args)
    # print("args: ", args)
    model = get_detector_model(args)
    model = model(args, device).to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')

    if args.checkpoint:
        if args.last or args.best:
            ckpt_path = (
                f'{args.dir_result}/{args.project_name}'
                + f'/ckpts/best_{str(seed_num)}.pth'
            )

        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        logger.best_auc = checkpoint['score']
        start_epoch = checkpoint['epoch']
        del checkpoint
    else:
        logger.best_auc = 0
        start_epoch = 1

    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr_init, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr_init, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr = args.lr_init, weight_decay=args.weight_decay)
    elif args.optim == 'adam_lars':
        optimizer = optim.Adam(model.parameters(), lr = args.lr_init, weight_decay=args.weight_decay)
        optimizer = LARC(optimizer=optimizer, eps=1e-8, trust_coefficient=0.001)
    elif args.optim == 'sgd_lars':
        optimizer = optim.SGD(model.parameters(), lr=args.lr_init, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer = LARC(optimizer=optimizer, eps=1e-8, trust_coefficient=0.001)
    elif args.optim == 'adamw_lars':
        optimizer = optim.AdamW(model.parameters(), lr = args.lr_init, weight_decay=args.weight_decay)
        optimizer = LARC(optimizer=optimizer, eps=1e-8, trust_coefficient=0.001)

    one_epoch_iter_num = len(train_loader)
    print("Iterations per epoch: ", one_epoch_iter_num)
    iteration_num = args.epochs * one_epoch_iter_num

    if args.lr_scheduler == "CosineAnnealing":
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=args.t_0*one_epoch_iter_num, T_mult=args.t_mult, eta_max=args.lr_max, T_up=args.t_up*one_epoch_iter_num, gamma=args.gamma)
    elif args.lr_scheduler == "Single":
        scheduler = CosineAnnealingWarmUpSingle(optimizer, max_lr=args.lr_init * math.sqrt(args.batch_size), epochs=args.epochs, steps_per_epoch=one_epoch_iter_num, div_factor=math.sqrt(args.batch_size))

    model.train()
    iteration = 0
    logger.loss = 0

    start = time.time()
    pbar = tqdm(total=args.epochs, initial=0, bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
    for epoch in range(start_epoch, args.epochs+1):
        for train_batch in train_loader:
            train_x, train_y, seq_lengths, target_lengths, aug_list, signal_name_list = train_batch
            train_x, train_y = train_x.to(device), train_y.to(device)
            iteration += 1

            model, iter_loss = get_trainer(args, iteration, train_x, train_y, seq_lengths, target_lengths, model, logger, device, scheduler, optimizer, criterion, signal_name_list)
            logger.loss += np.mean(iter_loss)

            ### LOGGING
            if iteration % args.log_iter == 0:
                logger.log_tqdm(epoch, iteration, pbar)
                logger.log_scalars(iteration)

            ### VALIDATION
            if iteration % (one_epoch_iter_num//val_per_epochs) == 0:
                model.eval()
                logger.evaluator.reset()
                val_iteration = 0
                logger.val_loss = 0
                with torch.no_grad():
                    for batch in tqdm(val_loader):
                        val_x, val_y, seq_lengths, target_lengths, aug_list, signal_name_list = batch
                        val_x, val_y = val_x.to(device), val_y.to(device)
                        model, val_loss = get_trainer(args, iteration, val_x, val_y, seq_lengths, target_lengths, model, logger, device, scheduler, optimizer, criterion, signal_name_list, flow_type=args.test_type)
                        logger.val_loss += np.mean(val_loss)
                        val_iteration += 1

                    logger.log_val_loss(val_iteration, iteration)
                    logger.add_validation_logs(iteration)
                    logger.save(model, optimizer, iteration, epoch)
                model.train()
        pbar.update(1)

    model.eval()
    logger.evaluator.reset()
    with torch.no_grad():
        for test_batch in tqdm(test_loader, total=len(test_loader), bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}"):
            test_x, test_y, seq_lengths, target_lengths, aug_list, signal_name_list = test_batch
            test_x = test_x.to(device)

            ### Model Structures
            if args.task_type == "binary": 
                model, _ = sliding_window_v1(args, iteration, test_x, test_y, seq_lengths, 
                                            target_lengths, model, logger, device, scheduler,
                                            optimizer, criterion, signal_name_list=signal_name_list, flow_type="test")    # margin_test , test
            else:
                print("Selected trainer is not prepared yet...")
                exit(1)

    logger.test_result_only()
    list_of_test_results_per_seed.append(logger.test_results)
    logger.writer.close()

auc_list = []
apr_list = []
f1_list = []
tpr_list = []
tnr_list = []
os.system("echo  \'#######################################\'")
os.system("echo  \'##### Final test results per seed #####\'")
os.system("echo  \'#######################################\'")
for result, tpr, tnr in list_of_test_results_per_seed:    
    os.system(
        f"echo  \'seed_case:{str(result[0])} -- auc: {str(result[1])}, apr: {str(result[2])}, f1_score: {str(result[3])}, tpr: {str(tpr)}, tnr: {str(tnr)}\'"
    )

    auc_list.append(result[1])
    apr_list.append(result[2])
    f1_list.append(result[3])
    tpr_list.append(tpr)
    tnr_list.append(tnr)
os.system(
    f"echo  \'Total average -- auc: {str(np.mean(auc_list))}, apr: {str(np.mean(apr_list))}, f1_score: {str(np.mean(f1_list))}, tnr: {str(np.mean(tpr_list))}, tpr: {str(np.mean(tnr_list))}\'"
)

os.system(
    f"echo  \'Total std -- auc: {str(np.std(auc_list))}, apr: {str(np.std(apr_list))}, f1_score: {str(np.std(f1_list))}, tnr: {str(np.std(tpr_list))}, tpr: {str(np.std(tnr_list))}\'"
)
    
