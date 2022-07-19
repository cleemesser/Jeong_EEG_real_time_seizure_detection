#!/usr/bin/env python3

# Copyright (c) 2022, Hyewon Jeong, Kwanhyung Lee. All rights reserved.
#
# Licensed under the MIT License; 
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import shutil
import copy
import logging
import logging.handlers
from collections import OrderedDict
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, roc_auc_score, average_precision_score, f1_score, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from builder.utils.metrics import Evaluator


class Logger:
    def __init__(self, args):
        self.args = args
        self.args_save = copy.deepcopy(args)

        # Evaluator
        self.evaluator = Evaluator(self.args)

        # Checkpoint and Logging Directories
        self.dir_root = os.path.join(self.args.dir_result, self.args.project_name)
        self.dir_log = os.path.join(self.dir_root, 'logs')
        self.dir_save = os.path.join(self.dir_root, 'ckpts')
        self.log_iter = args.log_iter

        if args.reset and os.path.exists(self.dir_root):
            shutil.rmtree(self.dir_root, ignore_errors=True)
        if not os.path.exists(self.dir_root):
            os.makedirs(self.dir_root)
        if not os.path.exists(self.dir_save):
            os.makedirs(self.dir_save)
        elif os.path.exists(
            os.path.join(self.dir_save, f'last_{str(args.seed)}.pth')
        ) and os.path.exists(self.dir_log):
            shutil.rmtree(self.dir_log, ignore_errors=True)
        if not os.path.exists(self.dir_log):
            os.makedirs(self.dir_log)

        # Tensorboard Writer
        self.writer = SummaryWriter(logdir=self.dir_log, flush_secs=60)

        # Log variables
        self.loss = 0
        self.val_loss = 0
        self.best_auc = 0
        self.best_iter = 0
        self.best_result_so_far = np.array([])
        self.best_results = []

        # results
        self.test_results = {}

        self.pred_results = []
        self.ans_results = []


    def log_tqdm(self, epoch, step, pbar):
        tqdm_log = f"Epochs: {str(epoch)}, Iteration: {str(step)}, Loss: {str(self.loss / step)}"

        pbar.set_description(tqdm_log)
        
    def log_scalars(self, step):
        self.writer.add_scalar('train/loss', self.loss / step, global_step=step)
    
    def log_lr(self, lr, step):
        self.writer.add_scalar('train/lr', lr, global_step=step)

    def log_val_loss(self, val_step, step):
        self.writer.add_scalar('val/loss', self.val_loss / val_step, global_step=step)

    def add_validation_logs(self, step):
        if self.args.task_type == "binary":
            result, tpr, fnr, tnr, fpr = self.evaluator.performance_metric_binary()
            auc = result[0]
            os.system("echo  \'##### Current Validation results #####\'")
            os.system(
                f"echo  \'auc: {str(result[0])}, apr: {str(result[1])}, f1_score: {str(result[2])}\'"
            )

            os.system(
                f"echo  \'tpr: {str(tpr)}, fnr: {str(fnr)}, tnr: {str(tnr)}, fpr: {str(fpr)}\'"
            )


            self.writer.add_scalar('val/auc', result[0], global_step=step)
            self.writer.add_scalar('val/apr', result[1], global_step=step)
            self.writer.add_scalar('val/f1', result[2], global_step=step)
            self.writer.add_scalar('val/tpr', tpr, global_step=step)
            self.writer.add_scalar('val/fnr', fnr, global_step=step)
            self.writer.add_scalar('val/tnr', tnr, global_step=step)
            self.writer.add_scalar('val/fpr', fpr, global_step=step)

            if self.best_auc < auc:
                self.best_iter = step
                self.best_auc = auc
                self.best_result_so_far = result
                self.best_results = [tpr, fnr, tnr, fpr]

            os.system("echo  \'##### Best Validation results in history #####\'")
            os.system(
                f"echo  \'auc: {str(self.best_result_so_far[0])}, apr: {str(self.best_result_so_far[1])}, f1_score: {str(self.best_result_so_far[2])}\'"
            )

            os.system(
                f"echo  \'tpr: {str(self.best_results[0])}, fnr: {str(self.best_results[1])}, tnr: {str(self.best_results[2])}, fpr: {str(self.best_results[3])}\'"
            )


        else:
            result, result_aucs, result_aprs, result_f1scores, tprs, fnrs, tnrs, fprs, fdrs, ppvs = self.evaluator.performance_metric_multi()

            multi_weighted_auc = result[0]
            multi_unweighted_auc = result[1]
            multi_weighted_apr = result[2]
            multi_unweighted_apr = result[3]
            multi_weighted_f1_score = result[4]
            multi_unweighted_f1_score = result[5]

            os.system("echo  \'##### Current Validation results #####\'")
            os.system(
                f"echo  \'multi_weighted: auc: {str(result[0])}, apr: {str(result[2])}, f1_score: {str(result[4])}\'"
            )

            os.system(
                f"echo  \'multi_unweighted: auc: {str(result[1])}, apr: {str(result[3])}, f1_score: {str(result[5])}\'"
            )

            os.system("echo  \'##### Each class Validation results #####\'")
            seizure_list = self.args.num_to_seizure_items
            results = [
                f"Label:{seizure} auc:{str(result_aucs[idx])} apr:{str(result_aprs[idx])} f1:{str(result_f1scores[idx])} tpr:{str(tprs[idx])} fnr:{str(fnrs[idx])} tnr:{str(tnrs[idx])} fpr:{str(fprs[idx])} fdr:{str(fdrs[idx])} ppv:{str(ppvs[idx])}"
                for idx, seizure in enumerate(seizure_list)
            ]

            for i in results:
                os.system(f"echo  \'{i}\'")

            self.writer.add_scalar('val/multi_weighted_auc', multi_weighted_auc, global_step=step)
            self.writer.add_scalar('val/multi_weighted_apr', multi_weighted_apr, global_step=step)
            self.writer.add_scalar('val/multi_weighted_f1_score', multi_weighted_f1_score, global_step=step)
            self.writer.add_scalar('val/multi_unweighted_auc', multi_unweighted_auc, global_step=step)
            self.writer.add_scalar('val/multi_unweighted_apr', multi_unweighted_apr, global_step=step)
            self.writer.add_scalar('val/multi_unweighted_f1_score', multi_unweighted_f1_score, global_step=step)

            if self.best_auc < multi_weighted_auc:
                self.best_iter = step
                self.best_auc = multi_weighted_auc
                self.best_result_so_far = result
                self.best_results = results

            os.system("echo  \'##### Best Validation results in history #####\'")
            os.system(
                f"echo  \'multi_weighted: auc: {str(self.best_result_so_far[0])}, apr: {str(self.best_result_so_far[2])}, f1_score: {str(self.best_result_so_far[4])}\'"
            )

            os.system(
                f"echo  \'multi_unweighted: auc: {str(self.best_result_so_far[1])}, apr: {str(self.best_result_so_far[3])}, f1_score: {str(self.best_result_so_far[5])}\'"
            )

            for i in self.best_results:
                os.system(f"echo  \'{i}\'")

        self.writer.flush()

    def save(self, model, optimizer, step, epoch, last=None):
        ckpt = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'best_step': step, 'last_step' : last, 'score' : self.best_auc, 'epoch' : epoch}

        if step == self.best_iter:
            self.save_ckpt(ckpt, f'best_{str(self.args.seed)}.pth')

        if last:
            self.evaluator.get_attributions()
            self.save_ckpt(ckpt, f'last_{str(self.args.seed)}.pth')
    
    def save_ckpt(self, ckpt, name):
        torch.save(ckpt, os.path.join(self.dir_save, name))

    def test_result_only(self):

        if self.args.task_type in ["binary", "binary_noslice"]:
            result, tpr, fnr, tnr, fpr = self.evaluator.performance_metric_binary()

            os.system("echo  \'##### Test results #####\'")
            os.system(
                f"echo  \'auc: {str(result[0])}, apr: {str(result[1])}, f1_score: {str(result[2])}\'"
            )

            os.system(
                f"echo  \'tpr: {str(tpr)}, fnr: {str(fnr)}, tnr: {str(tnr)}, fpr: {str(fpr)}\'"
            )


            # self.test_results.append("seed_case:{} -- auc: {}, apr: {}, f1_score: {}".format(str(self.args.seed), str(result[0]), str(result[1]), str(result[2])))
            self.test_results = [
                [self.args.seed, result[0], result[1], result[2]],
                tpr,
                tnr,
            ]

        else:
            result, result_aucs, result_aprs, result_f1scores, tprs, fnrs, tnrs, fprs, fdrs, ppvs = self.evaluator.performance_metric_multi()

            multi_weighted_auc = result[0]
            multi_unweighted_auc = result[1]
            multi_weighted_apr = result[2]
            multi_unweighted_apr = result[3]
            multi_weighted_f1_score = result[4]
            multi_unweighted_f1_score = result[5]

            os.system("echo  \'##### Test results #####\'")
            os.system(
                f"echo  \'multi_weighted: auc: {str(result[0])}, apr: {str(result[2])}, f1_score: {str(result[4])}\'"
            )

            os.system(
                f"echo  \'multi_unweighted: auc: {str(result[1])}, apr: {str(result[3])}, f1_score: {str(result[5])}\'"
            )

            os.system("echo  \'##### Each class Validation results #####\'")
            seizure_list = self.args.diseases_to_train
            results = [
                f"Label:bckg auc:{str(result_aucs[0])} apr:{str(result_aprs[0])} f1:{str(result_f1scores[0])} tpr:{str(tprs[0])} fnr:{str(fnrs[0])} tnr:{str(tnrs[0])} fpr:{str(fprs[0])} fdr:{str(fdrs[0])} ppv:{str(ppvs[0])}"
            ]

            results.extend(
                f"Label:{seizure} auc:{str(result_aucs[idx+1])} apr:{str(result_aprs[idx+1])} f1:{str(result_f1scores[idx+1])} tpr:{str(tprs[idx+1])} fnr:{str(fnrs[idx+1])} tnr:{str(tnrs[idx+1])} fpr:{str(fprs[idx+1])} fdr:{str(fdrs[idx+1])} ppv:{str(ppvs[idx+1])}"
                for idx, seizure in enumerate(seizure_list)
            )

            for i in results:
                os.system(f"echo  \'{i}\'")


        