# -*- coding: utf-8 -*-
# file: main_asp.py
# author: xunna <xunan2015@ia.ac.cn>
# Copyright (C) 2018. All Rights Reserved.
# Requirement: torch 0.4.0

from data_utils import ABSADatesetReader, ZOLDatesetReader
from data_utils_masad import MasadDatasetReader
from data_utils_trc import TrcDatasetReader

import time
import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from visdom import Visdom
from sklearn.metrics import accuracy_score
from sklearn import metrics
from torchvision.models import alexnet, resnet18, resnet50, inception_v3
from models.lstm import LSTM
from models.ian import IAN, IAN2m
from models.memnet import MemNet
from models.ram import RAM, RAM2m
from models.mimn import MIMN
from models.td_lstm import TD_LSTM
from models.cabasc import Cabasc
from models.memnet2 import MemNet2
from models.ae_lstm import AELSTM
from models.joint_model import JointModel_v1, JointModel_v2, JointModel_simplified
from models.end2end_lstm import End2End_LSTM

# Multi-Joint-Model complete version
from models.multi_joint_model import MultiJointModel_complete_v2
from models.feat_filter_model import FeatFilterModel

# ACD ablation model
from models.multi_joint_model import MultiJointModel_temp8, MultiJointModel_temp8_GlobalImgFeat, MultiJointModel_ACD_Ablation_GlobalImgFeat, MultiJointModel_ACD_Ablation_LocalImgFeat

# ASC ablation model
from models.multi_joint_model import MultiJointModel_temp9_modified_v2, MultiJointModel_temp9_modified_v2_GlobalImgFeat, MultiJointModel_temp9_modified_v2_SceneImgFeat, MultiJointModel_temp9_modified_v2_OnlyTextFeat

from my_focal_loss import focal_loss

np.random.seed(1337)  # for reproducibility

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        print('> training arguments:')
        for arg in vars(opt):
            print('>>> {0}: {1}'.format(arg, getattr(opt, arg)))
        
        # 根据加载的数据集创建对应的DataLoader
        if opt.dataset in ['restaurant', 'laptop']:
            self.my_dataset = ABSADatesetReader(dataset=opt.dataset, embed_dim=opt.embed_dim, max_seq_len=opt.max_seq_len)
            self.train_data_loader = DataLoader(dataset=self.my_dataset.train_data, batch_size=opt.batch_size, shuffle=True)
            self.dev_data_loader = DataLoader(dataset=self.my_dataset.test_data, batch_size=len(self.my_dataset.test_data), shuffle=False)
            self.test_data_loader = DataLoader(dataset=self.my_dataset.test_data, batch_size=len(self.my_dataset.test_data), shuffle=False)
        elif opt.dataset in ['zol_cellphone', 'zol_cellphone_zhoutao', 'zol_cellphone_zhoutao_ACD', 'zol_cellphone_grouped_label', 'zol_cellphone_zhoutao_grouped_label']:
            self.my_dataset = ZOLDatesetReader(dataset=opt.dataset, embed_dim=opt.embed_dim, max_seq_len=opt.max_seq_len, cnn_model_name=opt.cnn_model_name)
            self.train_data_loader = DataLoader(dataset=self.my_dataset.train_data, batch_size=opt.batch_size, shuffle=True)
            self.dev_data_loader = DataLoader(dataset=self.my_dataset.dev_data, batch_size=len(self.my_dataset.dev_data), shuffle=False)
            self.test_data_loader = DataLoader(dataset=self.my_dataset.test_data, batch_size=len(self.my_dataset.test_data), shuffle=False)
        elif opt.dataset == 'MASAD':
            self.my_dataset = MasadDatasetReader(dataset=opt.dataset, embed_dim=opt.embed_dim, cnn_model_name=opt.cnn_model_name, expanded=opt.expanded, fine_grain_img=opt.fine_grain_img, scene_level_img=opt.scene_level_img)
            self.train_data_loader = DataLoader(dataset=self.my_dataset.train_data, batch_size=opt.batch_size, shuffle=True)
            self.dev_data_loader = DataLoader(dataset=self.my_dataset.dev_data, batch_size=opt.batch_size, shuffle=False)
            self.test_data_loader = DataLoader(dataset=self.my_dataset.test_data, batch_size=opt.batch_size, shuffle=False)
            self.aspect2idx = self.my_dataset.aspect2idx
            self.idx2aspect = self.my_dataset.idx2aspect
            self.word2idx = self.my_dataset.word2idx
            self.idx2word = self.my_dataset.idx2word
        elif opt.dataset == 'TRC':
            self.my_dataset = TrcDatasetReader(dataset=opt.dataset, embed_dim=opt.embed_dim, cnn_model_name=opt.cnn_model_name)
            self.train_data_loader = DataLoader(dataset=self.my_dataset.train_data, batch_size=opt.batch_size, shuffle=True)
            self.dev_data_loader = DataLoader(dataset=self.my_dataset.dev_data, batch_size=opt.batch_size, shuffle=False)
            self.test_data_loader = DataLoader(dataset=self.my_dataset.test_data, batch_size=opt.batch_size, shuffle=False)
            self.word2idx = self.my_dataset.word2idx
            self.idx2word = self.my_dataset.idx2word
            
        self.idx2word = self.my_dataset.idx2word
        #self.writer = SummaryWriter(log_dir=opt.logdir)
        
        if opt.dataset != 'MASAD' or (opt.dataset == 'MASAD' and opt.shared_emb == 1):  # shared word embedding
            if opt.dataset == 'MASAD' and opt.model_name in ['joint_model', 'multi_joint_model']:
                self.model = opt.model_class(self.my_dataset.embedding_matrix, self.aspect2idx, opt).to(opt.device)
            else:
                self.model = opt.model_class(self.my_dataset.embedding_matrix, opt).to(opt.device)
        elif opt.dataset == 'MASAD' and opt.shared_emb == 0:  # independent
            embedding_matrices = [self.my_dataset.embedding_matrix, self.my_dataset.aspect_embedding_matrix]
            if opt.model_name == ['joint_model', 'multi_joint_model']:
                self.model = opt.model_class(self.my_dataset.embedding_matrix, self.aspect2idx, opt).to(opt.device)
            else:
                self.model = opt.model_class(embedding_matrices, opt).to(opt.device)
        
        self.reset_parameters()

    def reset_parameters(self):  # 模型参数初始化
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            # print(p.shape)
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
                if len(p.shape) > 1:
                     if p.shape[0] < 10000:  # do not change embedding layer
                         self.opt.initializer(p)
                    # self.opt.initializer(p)
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))

    def get_metrics(self, truth, pred):
        assert len(truth) == len(pred)

        y_true = [t.item() for t in truth]  # [tensor, ..] --> [int, ..]; Eg: [tensor(5), tensor(6)] --> [5, 6]
        y_pred = pred # [int, ..]
        
        acc = accuracy_score(y_true, y_pred) 
        f1 = metrics.f1_score(y_true, y_pred, average='weighted')
        return acc * 100, f1 * 100

    def get_accuracy(self, truth, pred):
        assert len(truth) == len(pred)
        y_true = truth
        y_pred = pred
        acc = accuracy_score(y_true, y_pred)
        return acc * 100
    
    def get_conf_matrix(self, truth, pred):
        assert len(truth) == len(pred)
        po_dim = self.opt.polarities_dim
        conf_matrix = [[0 for i in range(po_dim)] for i in range(po_dim)]
        
        for i in range(len(truth)):
            conf_matrix[int(truth[i])][int(pred[i])] += 1
        
        class_num = []  # 各类别样本数量
        class_right_num = []  # 各类别正确样本数量
        acc = []  # 各类别准确率
        
        for i in range(len(conf_matrix)):
            class_num.append(sum(conf_matrix[i]))
            class_right_num.append(conf_matrix[i][i])
            if class_num[i] > 0:
                acc.append((class_right_num[i]/class_num[i])*100)
            else:
                acc.append(-1)
        
        print('cofusion matrix:')
        for i in range(po_dim):
            print(conf_matrix[i])
        
        print('\nacc for each class: ')
        for i in range(po_dim):
            print('class %d: %f' % (i+1, acc[i]))
        print('\n')
        return
    
    def get_conf_matrix_normal(self, truth, pred, class_dim):
        assert len(truth) == len(pred)
        conf_matrix = [[0 for i in range(class_dim)] for i in range(class_dim)]
        
        for i in range(len(truth)):
            conf_matrix[int(truth[i])][int(pred[i])] += 1

        return conf_matrix

    def findword(self, idxs, idx2word):
        text = ""
        idxs = idxs.cpu().numpy()
        for id in idxs:
            if id in idx2word:
                text += idx2word[id]+" "
            else:
                text += str(id) + " "
        return text

    def run(self):
        # Loss and Optimizer
        dtw = time.strftime("%Y-%m-%d-%H-%M", time.localtime(int(time.time())))
        
        best_dev_acc = 0.0  # 验证集最优acc
        best_dev_f1 = 0.0  # 验证集最优f1
        no_up = 0  # 验证集f1连续未提升的轮次，>=10即早停
        
        loss_function = nn.CrossEntropyLoss()  # CE
        #loss_function = focal_loss(alpha = [0.99999 for i in range(self.opt.polarities_dim)], num_classes = self.opt.polarities_dim)  # focal loss
        
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(params, lr=self.opt.learning_rate)
        
        
        for epoch in range(self. opt.num_epoch):
            start = time.time()
            # print('>' * 100)
            tra_loss, tra_acc, tra_f1 = self.train_epoch(loss_function, optimizer, epoch)
            # print('now best dev f1:', best_dev_f1)
            dev_loss, dev_acc, dev_f1, dev_truth, dev_pred = self.evaluate(loss_function, 'dev')

            end = time.time()
            epoch_dt = end - start

            print('epoch: %d done, %d s! Train avg_loss:%g , acc:%g, f1:%g, Dev loss:%g acc:%g f1:%g' % (epoch, epoch_dt, tra_loss, tra_acc, tra_f1, dev_loss, dev_acc, dev_f1))
            #print('epoch: %d, Dev Confusion Matrix as follow:' % epoch)
            #self.get_conf_matrix(dev_truth, dev_pred)

            if dev_f1 > best_dev_f1:
                if not os.path.exists('./checkpoint'):
                    os.mkdir('./checkpoint')
                best_dev_f1 = dev_f1
                os.system('rm ./checkpoint/'+self.opt.dataset+'_'+self.opt.model_name+'_best_acc_*'+dtw+'.models')
                test_loss, test_acc, test_f1, truth, pred = self.evaluate(loss_function, 'test')
                print('New Best Test acc, f1:', test_acc, test_f1)
                #self.get_conf_matrix(truth, pred)  # 输出混淆矩阵信息
                torch.save(self.model.state_dict(), './checkpoint/'+self.opt.dataset+'_'+self.opt.model_name+'_best_acc_%.4g_f1_%.4g' % (test_acc, test_f1) +'_time_'+dtw+ '.models')
                no_up = 0
            else:
                no_up += 1
                if no_up >= self.opt.early_stop:
                    exit()
    
    def run_acsa(self):
        '''
        for ACSA-task
        '''
        # Loss and Optimizer
        dtw = time.strftime("%Y-%m-%d-%H-%M", time.localtime(int(time.time())))
        
        best_dev_acsa_acc = 0.0  # 验证集最优的ACSA任务acc
        best_dev_acsa_f1 = 0.0  # 验证集最优的ACSA任务f1
        no_up = 0  # 验证集f1连续未提升的轮次，>=10即早停
        
        # params <reduction> is default to 'mean', meaning that the loss_function would return the average value of all sample losses.
        # But the ASC-task Loss is not responsible for all samples, so we set params <reduction> to 'none' to get an sample loss array.
        loss_function = nn.CrossEntropyLoss(reduction='none')
        # loss_function = focal_loss(alpha = [0.99999 for i in range(self.opt.polarities_dim)], num_classes = self.opt.polarities_dim)  # focal loss
        
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(params, lr=self.opt.learning_rate)
        
        
        for epoch in range(self. opt.num_epoch):
            start = time.time()
            print('>' * 100)
            tra_loss, tra_acsa_acc, tra_acsa_f1, tra_acd_acc, tra_acd_f1, tra_asc_acc = self.train_epoch_acsa(loss_function, optimizer, epoch)
            dev_loss, dev_acsa_acc, dev_acsa_f1, dev_acd_acc, dev_acd_f1, dev_asc_acc = self.evaluate_acsa(loss_function, 'dev')
            # print('now best dev f1:', best_dev_f1)

            end = time.time()
            epoch_dt = end - start

            print('epoch: %d done, %d s!' % (epoch, epoch_dt))
            print('Train avg_loss:%g, ACSA_acc:%g, ACSA_f1:%g' % (tra_loss, tra_acsa_acc, tra_acsa_f1))
            print('Train ACD_acc:%g, ACD_f1:%g, ASC_acc:%g' % (tra_acd_acc, tra_acd_f1, tra_asc_acc))
            print('Dev avg_loss:%g, ACSA_acc:%g, ACSA_f1:%g' % (dev_loss, dev_acsa_acc, dev_acsa_f1))
            print('Dev ACD_acc:%g, ACD_f1:%g, ASC_acc:%g' % (dev_acd_acc, dev_acd_f1, dev_asc_acc))
            #print('epoch: %d, Dev Confusion Matrix as follow:' % epoch)
            #self.get_conf_matrix(dev_truth, dev_pred)

            if dev_acsa_f1 > best_dev_acsa_f1:
                if not os.path.exists('./checkpoint'):
                    os.mkdir('./checkpoint')
                best_dev_acsa_f1 = dev_acsa_f1
                os.system('rm ./checkpoint/'+self.opt.dataset+'_'+self.opt.model_name+'_best_acc_*'+dtw+'.models')
                test_loss, test_acsa_acc, test_acsa_f1, test_acd_acc, test_acd_f1, test_asc_acc = self.evaluate_acsa(loss_function, 'test')
                print('New Best Test ACSA_acc:%g, ACSA_f1:%g' % (test_acsa_acc, test_acsa_f1))
                print('New Best Test ACD_acc:%g, ACD_f1:%g, ASC_acc:%g' % (test_acd_acc, test_acd_f1, test_asc_acc))
                # self.get_conf_matrix(truth, pred)  # 输出混淆矩阵信息
                torch.save(self.model.state_dict(), './checkpoint/'+self.opt.dataset+'_'+self.opt.model_name+'_best_acc_%.4g_f1_%.4g' % (test_acsa_acc, test_acsa_f1) +'_time_'+dtw+ '.models')
                no_up = 0
            else:
                no_up += 1
                if no_up >= self.opt.early_stop:
                    exit()

    def test1(self):
        # Loss and Optimizer
        loss_function = nn.CrossEntropyLoss()
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        # switch models to evaluation mode
        self.model.eval()
        print(self.model)
        avg_loss = 0.0
        truth_res = []
        pred_res = []

        with torch.no_grad():  # 避免梯度累积
            for batch, sample_batched in enumerate(self.dev_data_loader):
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = sample_batched['polarity'].to(self.opt.device)  # torch.tensor([label, label, label, ...]), (batch_size, )
                truth_res += list(targets.data)
                outputs = self.model(inputs) # (batch_size, polarity_dim)
                pred_label = outputs.data.max(1)[1].cpu().numpy()
                pred_res += [x for x in pred_label]
                loss = loss_function(outputs, targets)
                avg_loss += loss.item()
                break
            
            '''
            # 标签target和预测结果pred，都是从0开始
            truch_res = [tensor(1), tensor(2), tensor(1), tensor(5), ...]
            pred_res = [1,2,1,5,...]
            '''
            avg_loss /= len(self.dev_data_loader)
            acc = self.get_accuracy(truth_res, pred_res)
            return avg_loss, acc

    
    def test2(self, modelname):
        # Loss and Optimizer
        loss_function = nn.CrossEntropyLoss()
        params = filter(lambda p: p.requires_grad, self.model.parameters())

        self.model.load_state_dict(torch.load('./checkpoint/'+modelname))

        # switch models to evaluation mode
        self.model.eval()
        avg_loss = 0.0
        truth_res = []
        pred_res = []
        text_data = []
        img_data = []
        aspect_data = []
        score_texts = []
        score_imgs = []

        with torch.no_grad():  # 避免梯度累积
            for batch, sample_batched in enumerate(self.dev_data_loader):
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = sample_batched['polarity'].to(self.opt.device)
                truth_res += list(targets.data)
                outputs, score_text, score_img = self.model(inputs)
                pred_label = outputs.data.max(1)[1].cpu().numpy()
                pred_res += [x for x in pred_label]
                text_data += list(inputs[0].data)
                aspect_data += list(inputs[1].data)
                img_data += list(inputs[2].data)
                score_texts += list(score_text.data)
                score_imgs += list(score_img.data)

                break


            for i in range(len(truth_res)):
                print('sample======',i )

                print(self.findword(text_data[i]) )
                print(self.findword(aspect_data[i]) )
                # print(score_texts[i])
                # print(score_imgs[i])
                # print(truth_res[i], pred_res[i])



            acc = self.get_accuracy(truth_res, pred_res)
            return avg_loss, acc
    
    
    def test3(self, modelname):
        # Loss and Optimizer
        loss_function = nn.CrossEntropyLoss()
        params = filter(lambda p: p.requires_grad, self.model.parameters())

        self.model.load_state_dict(torch.load('./checkpoint/'+modelname))

        # switch models to evaluation mode
        self.model.eval()
        avg_loss = 0.0
        truth_res = []
        pred_res = []
        text_data = []
        img_data = []
        aspect_data = []
        score_texts = []
        score_imgs = []

        with torch.no_grad():  # 避免梯度累积
            for batch, sample_batched in enumerate(self.dev_data_loader):
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = sample_batched['polarity'].to(self.opt.device)
                truth_res += list(targets.data)
                outputs = self.model(inputs)
                pred_label = outputs.data.max(1)[1].cpu().numpy()
                pred_res += [x for x in pred_label]

                break


            for i in range(len(truth_res)):
                print('sample======',i )

                print(truth_res[i], pred_res[i])


            self.get_conf_matrix(truth_res, pred_res)
            acc = self.get_accuracy(truth_res, pred_res)
            print('total acc: ', acc)
            return avg_loss, acc
        
        
    def test_acd(self, modelname):
        '''
        ACD BadCase Analysis    
        '''
        # Loss and Optimizer
        loss_function = nn.CrossEntropyLoss()
        params = filter(lambda p: p.requires_grad, self.model.parameters())

        self.model.load_state_dict(torch.load('./checkpoint/'+modelname))

        # switch models to evaluation mode
        self.model.eval()
        avg_loss = 0.0
        ACD_truth_res = []
        ACD_pred_res = []
        text_data = []  # sample text
        aspect_data = []  # sample aspect
        sample_ids = []  # sample id

        with torch.no_grad():  # 避免梯度累积
            for batch, sample_batched in enumerate(self.test_data_loader):
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                
                ACSA_targets = sample_batched['acsa_label'].to(self.opt.device) # 0=negative, 1=positive, 2=irrelevant
                ACD_targets = sample_batched['acd_label'].to(self.opt.device)  # 0=irrelevant, 1=relevant
                ASC_targets = sample_batched['asc_label'].to(self.opt.device)  # 0=negative, 1=positive
                
                ACD_outputs, ASC_outputs = self.model(inputs)
                ACD_pred_label = list(ACD_outputs.data.max(1)[1].cpu().numpy())

                ACD_truth_res += ACD_targets.tolist()
                ACD_pred_res += ACD_pred_label
                
                text_data += list(inputs[0].data)
                aspect_data += list(sample_batched['aspect_id'].to(self.opt.device).data)
                sample_ids += sample_batched['id'].to(self.opt.device).tolist()
            
            print('Error Type-A: predict Irrelevant to Relevant')
            print('Error Type-B: predict Relevant to Irrelevant')
            answer = input('Start BadCase Study for Error Type-A? yes or no: ')
            if answer == 'yes':
                for i in range(len(ACD_truth_res)):
                    if ACD_truth_res[i] != ACD_pred_res[i] and ACD_truth_res[i] == 0:
                        print('sample======',i )
                        print('Text: ', self.findword(text_data[i], self.idx2word))
                        aspect = (aspect_data[i] + 1).unsqueeze(dim=0)  # dim=0 --> dim=1
                        print('Aspect: ', self.findword(aspect, self.idx2aspect))
                        print('Sample ID: ', sample_ids[i])
                        print('Label: ', ACD_truth_res[i])
                        print('Prediction: ', ACD_pred_res[i])
                        ans = input('Continue checking? yes or no: ')
                        if ans == 'no':
                            break
            answer = input('Start BadCase Study for Error Type-B? yes or no: ')
            if answer == 'yes':
                for i in range(len(ACD_truth_res)):
                    if ACD_truth_res[i] != ACD_pred_res[i] and ACD_truth_res[i] == 1:
                        print('sample======',i )
                        print('Text: ', self.findword(text_data[i], self.idx2word))
                        aspect = (aspect_data[i] + 1).unsqueeze(dim=0)  # dim=0 --> dim=1
                        print('Aspect: ', self.findword(aspect, self.idx2aspect))
                        print('Sample ID: ', sample_ids[i])
                        print('Label: ', ACD_truth_res[i])
                        print('Prediction: ', ACD_pred_res[i])
                        ans = input('Continue checking? yes or no: ')
                        if ans == 'no':
                            break
            
            # 1. Global Confusion Matrix and Acc
            conf_matrix = self.get_conf_matrix_normal(ACD_truth_res, ACD_pred_res, class_dim=2)
            acc = self.get_accuracy(ACD_truth_res, ACD_pred_res)
            print('\nglobal cofusion matrix:')
            for i in range(len(conf_matrix)):
                print(conf_matrix[i])
            print('\ntotal acc: ', acc)
            
            # 2. Aspect Statistics
            aspect_total_nums = [0 for i in range(57)]
            aspect_gold_nums = [0 for i in range(57)]
            for i in range(len(ACD_truth_res)):
                id = aspect_data[i].item()
                aspect_total_nums[id] += 1
                if ACD_truth_res[i] == ACD_pred_res[i]:
                    aspect_gold_nums[id] += 1
            aspect_stat = {}
            for i in range(57):
                aspect_name = self.findword(torch.tensor(i+1).unsqueeze(dim=0), self.idx2aspect)
                aspect_acc = aspect_gold_nums[i] / aspect_total_nums[i]
                aspect_stat[aspect_name] = [aspect_acc, aspect_total_nums[i]]
            print('\naspect statistics:')
            print(aspect_stat)
            
            # 3. Text Length Check
            error_A_len, error_A_num = 0, 0  # predict irrelevant to relevant
            error_B_len, error_B_num = 0, 0  # predict relevant to irrelevant
            total_len = 0
            for i in range(len(ACD_truth_res)):
                curr_len = torch.sum(text_data[i] != 0, dim=0)
                total_len += curr_len
                if ACD_truth_res[i] != ACD_pred_res[i]:
                    if ACD_truth_res[i] == 0:
                        error_A_len += curr_len
                        error_A_num += 1
                    else:
                        error_B_len += curr_len
                        error_B_num += 1
            gold_len, gold_num = total_len - error_A_len - error_B_len, len(ACD_truth_res) - error_A_num - error_B_num
            error_A_len = error_A_len / error_A_num
            error_B_len = error_B_len / error_B_num
            total_len = total_len / len(ACD_truth_res)
            gold_len = gold_len / gold_num
            print('\nText Avg Length: ', total_len)
            print('Gold Sample Length: ', gold_len)
            print('Error Type-A Avg Length: ', error_A_len)
            print('Error Type-B Avg Length: ', error_B_len)
            
            return 


    def evaluate(self, loss_function, name='dev'):
        # switch models to evaluation mode
        self.model.eval()
        avg_loss = 0.0
        truth_res = []
        pred_res = []

        if name == 'dev':
            evaluate_data = self.dev_data_loader
        elif name == 'test':
            evaluate_data = self.test_data_loader

        with torch.no_grad():  # 避免梯度累积
            for batch, sample_batched in enumerate(evaluate_data):
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                if self.opt.dataset == 'TRC':
                    targets = sample_batched['image_label'].to(self.opt.device)
                else:
                    targets = sample_batched['polarity'].to(self.opt.device)
                truth_res += list(targets.data)
                outputs = self.model(inputs)
                pred_label = outputs.data.max(1)[1].cpu().numpy()
                pred_res += [x for x in pred_label]
                loss = loss_function(outputs, targets)
                avg_loss += loss.item()

            avg_loss /= len(evaluate_data)
            acc, f1 = self.get_metrics(truth_res, pred_res)
            
            return avg_loss, acc, f1, truth_res, pred_res
    

    def evaluate_acsa(self, loss_function, name='dev'):
        '''
        for ACSA-task
        '''
        # switch models to evaluation mode
        self.model.eval()
        avg_loss = 0.0
        ACSA_truth_res, ACD_truth_res, ASC_truth_res = [], [], []
        ACSA_pred_res, ACD_pred_res, ASC_pred_res = [], [], []
        
        if name == 'dev':
            evaluate_data = self.dev_data_loader
        elif name == 'test':
            evaluate_data = self.test_data_loader

        with torch.no_grad():  # 避免梯度累积
            for batch, sample_batched in enumerate(evaluate_data):
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                ACSA_targets = sample_batched['acsa_label'].to(self.opt.device) # 0=negative, 1=positive, 2=irrelevant
                ACD_targets = sample_batched['acd_label'].to(self.opt.device)  # 0=irrelevant, 1=relevant
                ASC_targets = sample_batched['asc_label'].to(self.opt.device)  # 0=negative, 1=positive         
                
                if self.opt.model_type == 'aod':  # Add-One-Dimension
                    # loss
                    outputs = self.model(inputs)  # (batch, 3)
                    loss = loss_function(outputs, ACSA_targets)  # (batch, )
                    loss = loss.mean()
                    # metrics
                    ACSA_pred_label = list(outputs.data.max(1)[1].cpu().numpy()) # 0=negative, 1=positive, 2=irrelevant
                    ACD_pred_label = [0 if label == 2 else 1 for label in ACSA_pred_label]
                    ASC_pred_label = [label if label != 2 else -1 for label in ACSA_pred_label]  # -1 --> no ASC-label
                elif self.opt.model_type == 'mtl':  # Multi-Task-Learning                    
                    outputs = self.model(inputs)
                    # main task loss
                    ACD_outputs, ASC_outputs = outputs[0], outputs[1]  # (batch, 2), (batch, 2)
                    acd_loss = loss_function(ACD_outputs, ACD_targets)  # (batch, )
                    asc_loss = loss_function(ASC_outputs, ASC_targets)  # (batch, )
                    asc_loss *= ACD_targets  #  Aspect-Null samples have no ASC-label, so have no ASC-loss
                    if self.opt.mtl_weight_mode == 'preset':  # preset asc_weight
                        loss = acd_loss.mean() + self.opt.asc_weight * asc_loss.mean()
                    elif self.opt.mtl_weight_mode == 'auto1':  # revise asc_weight as batch_size/aspect_real_num
                        aspect_real_num = ACD_targets.sum().item()
                        revise_asc_weight = self.opt.batch_size / aspect_real_num if aspect_real_num > 0 else 0.0
                        loss = acd_loss.mean() + revise_asc_weight * asc_loss.mean()
                    elif self.opt.mtl_weight_mode == 'auto2':  # normalize acd_loss and asc_loss to 1.0 respectively. refers to Jianlin Su
                        acd_weight = 1. / (acd_loss.mean().item() + 0.0001)
                        asc_weight = 1. / (asc_loss.mean().item() + 0.0001)
                        loss = acd_weight * acd_loss.mean() + asc_weight * asc_loss.mean()
                    # auxiliary task loss
                    if len(outputs) == 3:
                        Auxiliary_outputs = outputs[2]
                        auxiliary_loss = loss_function(Auxiliary_outputs, ACD_targets)
                        loss += 1.0 * auxiliary_loss.mean()
                    
                    # metrics
                    #ACD_pred_label = list(ACD_outputs.data.max(1)[1].cpu().numpy())  # 0=irrelevant, 1=relevant
                    ACD_pred_prob = nn.functional.softmax(ACD_outputs, dim=1).detach().cpu().numpy()  # (batch, 2)
                    ACD_pred_label = [1 if ACD_pred_prob[i][1] > self.opt.acd_thresold else 0 for i in range(len(ACD_pred_prob))]
                    ASC_pred_label = list(ASC_outputs.data.max(1)[1].cpu().numpy())  # 0=negative, 1=positive
                    ASC_pred_label = [ASC_pred_label[i] if ACD_pred_label[i] == 1 else -1 for i in range(len(ACD_pred_label))]
                    ACSA_pred_label = [label if label != -1 else 2 for label in ASC_pred_label]
                
                ACSA_truth_res += list(ACSA_targets.data)
                ACSA_pred_res += ACSA_pred_label
                
                ACD_truth_res += list(ACD_targets.data)
                ACD_pred_res += ACD_pred_label
                
                ASC_temp_truth = []
                ASC_temp_pred = []
                for i in range(len(ASC_pred_label)):
                    if ACD_targets[i].item() == 1 and ACD_pred_label[i] == 1:
                        ASC_temp_truth.append(ASC_targets[i].item())
                        ASC_temp_pred.append(ASC_pred_label[i])
                ASC_truth_res += ASC_temp_truth
                ASC_pred_res += ASC_temp_pred
                
                avg_loss += loss.item()
    
            avg_loss /= len(evaluate_data)
            ACSA_acc, ACSA_f1 = self.get_metrics(ACSA_truth_res, ACSA_pred_res)
            ACD_acc, ACD_f1 = self.get_metrics(ACD_truth_res, ACD_pred_res)
            ASC_acc = self.get_accuracy(ASC_truth_res, ASC_pred_res)
            
            return avg_loss, ACSA_acc, ACSA_f1, ACD_acc, ACD_f1, ASC_acc

    def train_epoch(self, loss_function, optimizer, i):
        # switch models to training mode, clear gradient accumulators
        # print('epoch:', i)
        self.model.train()
        avg_loss = 0.0
        count = 0
        truth_res = []
        pred_res = []
        for i_batch, sample_batched in enumerate(self.train_data_loader):
            optimizer.zero_grad()  # 清空之前的梯度结果，避免梯度累积
            inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
            if self.opt.dataset == 'TRC':
                targets = sample_batched['image_label'].to(self.opt.device)
            else:
                targets = sample_batched['polarity'].to(self.opt.device)
            truth_res += list(targets.data)
            outputs = self.model(inputs)
            pred_label = outputs.data.max(1)[1].cpu().numpy()
            pred_res += ([x for x in pred_label])
            loss = loss_function(outputs, targets)
            avg_loss += loss.item()
            count += 1
            if count % self.opt.log_step == 0:
                dt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time())))
                # print('"%s" iteration: [%d/%d] loss: %g' % (dt, count * self.opt.batch_size, len(self.my_dataset.train_data), loss.item()))
            loss.backward()
            optimizer.step()

        avg_loss /= len(self.train_data_loader)
        acc, f1 = self.get_metrics(truth_res, pred_res)
        return avg_loss, acc, f1


    def train_epoch_acsa(self, loss_function, optimizer, i):
        '''
        for ACSA-task, return loss, ACSA_acc, ACSA_f1, ACD_acc, ACD_f1, ASC_acc
        '''
        # switch models to training mode, clear gradient accumulators
        # print('epoch:', i)
        self.model.train()
        avg_loss = 0.0
        count = 0
        ACSA_truth_res, ACD_truth_res, ASC_truth_res = [], [], []
        ACSA_pred_res, ACD_pred_res, ASC_pred_res = [], [], []
        for i_batch, sample_batched in enumerate(self.train_data_loader):
            optimizer.zero_grad()  # 清空之前的梯度结果，避免梯度累积
            inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
            
            # XXX_targets is a tensor of list: tensor([label, label, label, ...]), label is a <int> value
            # targets = sample_batched['polarity'].to(self.opt.device)
            ACSA_targets = sample_batched['acsa_label'].to(self.opt.device) # 0=negative, 1=positive, 2=irrelevant
            ACD_targets = sample_batched['acd_label'].to(self.opt.device)  # 0=irrelevant, 1=relevant
            ASC_targets = sample_batched['asc_label'].to(self.opt.device)  # 0=negative, 1=positive         
            # XXX_truth_res is a list of tensor: [tensor(label), tensor(label), ...], label is a <int> value
            # truth_res += list(targets.data)
            #ACSA_truth_res += list(ACSA_targets.data)
            #ACD_truth_res += list(ACD_targets.data)
            #ASC_truth_res += list(ASC_targets.data)
            
            if self.opt.model_type == 'aod':  # Add-One-Dimension
                # loss
                outputs = self.model(inputs)  # (batch, 3)
                loss = loss_function(outputs, ACSA_targets)  # (batch, )
                loss = loss.mean()
                # metrics
                ACSA_pred_label = list(outputs.data.max(1)[1].cpu().numpy()) # 0=negative, 1=positive, 2=irrelevant
                ACD_pred_label = [0 if label == 2 else 1 for label in ACSA_pred_label]
                ASC_pred_label = [label if label != 2 else -1 for label in ACSA_pred_label]  # -1 --> no ASC-label
            elif self.opt.model_type == 'mtl':  # Multi-Task-Learning
                outputs = self.model(inputs)
                # main task loss
                ACD_outputs, ASC_outputs = outputs[0], outputs[1]  # (batch, 2), (batch, 2)
                acd_loss = loss_function(ACD_outputs, ACD_targets)  # (batch, )
                asc_loss = loss_function(ASC_outputs, ASC_targets)  # (batch, )
                asc_loss *= ACD_targets  #  Aspect-Null samples have no ASC-label, so have no ASC-loss
                if self.opt.mtl_weight_mode == 'preset':  # preset asc_weight
                    loss = acd_loss.mean() + self.opt.asc_weight * asc_loss.mean()
                elif self.opt.mtl_weight_mode == 'auto1':  # revise asc_weight as batch_size/aspect_real_num
                    aspect_real_num = ACD_targets.sum().item()
                    revise_asc_weight = self.opt.batch_size / aspect_real_num if aspect_real_num > 0 else 0.0
                    loss = acd_loss.mean() + revise_asc_weight * asc_loss.mean()
                elif self.opt.mtl_weight_mode == 'auto2':  # normalize acd_loss and asc_loss to 1.0 respectively. refers to Jianlin Su
                    acd_weight = 1. / (acd_loss.mean().item() + 0.0001)
                    asc_weight = 1. / (asc_loss.mean().item() + 0.0001)
                    loss = acd_weight * acd_loss.mean() + asc_weight * asc_loss.mean()
                # auxiliary task loss
                if len(outputs) == 3:
                    Auxiliary_outputs = outputs[2]
                    auxiliary_loss = loss_function(Auxiliary_outputs, ACD_targets)
                    loss += 1.0 * auxiliary_loss.mean()
                
                # metrics
                #ACD_pred_label = list(ACD_outputs.data.max(1)[1].cpu().numpy())  # 0=irrelevant, 1=relevant
                ACD_pred_prob = nn.functional.softmax(ACD_outputs, dim=1).detach().cpu().numpy()  # (batch, 2)
                ACD_pred_label = [1 if ACD_pred_prob[i][1] > self.opt.acd_thresold else 0 for i in range(len(ACD_pred_prob))]
                ASC_pred_label = list(ASC_outputs.data.max(1)[1].cpu().numpy())  # 0=negative, 1=positive
                ASC_pred_label = [ASC_pred_label[i] if ACD_pred_label[i] == 1 else -1 for i in range(len(ACD_pred_label))]
                ACSA_pred_label = [label if label != -1 else 2 for label in ASC_pred_label]
                
            # XXX_targets is a tensor of list: tensor([label, label, label, ...]), label is a <int> value
            ACSA_truth_res += list(ACSA_targets.data)
            ACSA_pred_res += ACSA_pred_label
            
            ACD_truth_res += list(ACD_targets.data)
            ACD_pred_res += ACD_pred_label
            
            ASC_temp_truth = []
            ASC_temp_pred = []
            for i in range(len(ASC_pred_label)):
                if ACD_targets[i].item() == 1 and ACD_pred_label[i] == 1:
                    ASC_temp_truth.append(ASC_targets[i].item())
                    ASC_temp_pred.append(ASC_pred_label[i])
            ASC_truth_res += ASC_temp_truth
            ASC_pred_res += ASC_temp_pred
            
            avg_loss += loss.item()
            count += 1
            
            if count % self.opt.log_step == 0:
                dt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time())))
                # print('"%s" iteration: [%d/%d] loss: %g' % (dt, count * self.opt.batch_size, len(self.my_dataset.train_data), loss.item()))
            loss.backward()
            optimizer.step()

        avg_loss /= len(self.train_data_loader)
        ACSA_acc, ACSA_f1 = self.get_metrics(ACSA_truth_res, ACSA_pred_res)
        ACD_acc, ACD_f1 = self.get_metrics(ACD_truth_res, ACD_pred_res)
        ASC_acc = self.get_accuracy(ASC_truth_res, ASC_pred_res)
        
        return avg_loss, ACSA_acc, ACSA_f1, ACD_acc, ACD_f1, ASC_acc


if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='multi_joint_model', type=str, help='ian, ram, lstm, ae_lstm, memnet, mimn, end2end_lstm, joint_model, multi_joint_model, feat_filter_model')
    parser.add_argument('--cnn_model_name', default='resnet50', type=str, help='resnet50, resnet18, vgg')
    parser.add_argument('--dataset', default='MASAD_expanded', type=str, help='restaurant, laptop, zol_cellphone, MASAD, MASAD_expanded, TRC')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--logdir', default='log', type=str)
    parser.add_argument('--embed_dim', default=300, type=int)  # word embedding dim
    parser.add_argument('--embed_dim_img', default=2048, type=int)  # resnet50 -- 2048; resnet18 -- 512
    parser.add_argument('--hidden_dim', default=150, type=int)
    parser.add_argument('--max_seq_len', default=320, type=int)  # Dataset-MultiZol -- 320; Dataset-MASAD -- 25; Dataset-TRC -- 14
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--device', default=0, type=str)
    parser.add_argument('--early_stop', default=10, type=int)
    parser.add_argument('--test_mode', default=False, type=bool)
    
    parser.add_argument('--shared_emb', default=1, type=int)  # 0 = False; 1 = True; default = 1
    parser.add_argument('--expanded', default=0, type=int)  # 0 = False; 1 = True; default = 0
    parser.add_argument('--fine_grain_img', default=0, type=int)  # 0 = False; 1 = True; default = 0
    parser.add_argument('--scene_level_img', default=0, type=int)  # 0 = False; 1 = True; default = 0
    parser.add_argument('--model_type', default='aod', type=str, help='aod, mtl')  # aod = Add-One-Dimension, mtl = Multi-Task-Learning
    
    parser.add_argument('--mtl_weight_mode', default='preset', type=str, help='preset, auto1, auto2')
    parser.add_argument('--asc_weight', default=5.0, type=float)  # default: 1.0
    parser.add_argument('--acd_thresold', default=0.5, type=float)  # default: 0.5
    
    parser.add_argument('--load_filter_module', default=False, type=bool)  # whether involve feat_filter_model into Multi-Joint-Model
    parser.add_argument('--pretrain_module_path', default='', type=str)  # path of pretrained feat_filter_model, default=''
    
    opt = parser.parse_args()
    if opt.dataset == 'zol_cellphone':
        opt.polarities_dim = 8
    if opt.dataset == 'zol_cellphone_zhoutao':  # Add-One-Dimension
        opt.polarities_dim = 9
    if opt.dataset == 'zol_cellphone_zhoutao_ACD':  # Aspect Detection Task
        opt.polarities_dim = 2
    if opt.dataset == 'zol_cellphone_grouped_label':  # initial Dataset, convert score labels into negative/neural/positive three types
        opt.polarities_dim = 3
    if opt.dataset == 'zol_cellphone_zhoutao_grouped_label':  # A-O-D Dataset, convert score labels into negative/neural/positive three types
        opt.polarities_dim = 4
    
    if opt.dataset == 'MASAD_expanded':
        opt.dataset = 'MASAD'
        opt.expanded = 1
        if opt.model_name in ['joint_model', 'multi_joint_model']:
            opt.model_type = 'mtl'
            opt.polarities_dim = 2
            opt.fine_grain_img = int(opt.model_name == 'multi_joint_model')  # involve fine-grained img features into ACD Network
            opt.scene_level_img = int(opt.model_name == 'multi_joint_model')  # involve scene-level img features into ASC Network
        elif opt.model_name in ['ae_lstm', 'end2end_lstm']:
            opt.polarities_dim = 3  # add one dimension
        opt.max_seq_len = 25
    elif opt.dataset == 'MASAD':
        opt.polarities_dim = 2
        opt.max_seq_len = 25
    elif opt.dataset == 'TRC':
        opt.polarities_dim = 2  # two-class
        opt.max_seq_len = 14
    
    if opt.model_name == 'multi_joint_model' and opt.load_filter_module is True:
            opt.pretrain_module_path = './checkpoint/trc_experiments/pretrain/' + 'TRC_feat_filter_model_best_acc_83.17_f1_82.96_time_2023-02-16-08-38.models'

    model_classes = {
        'lstm': LSTM,
        'ae_lstm': AELSTM,
        'td_lstm': TD_LSTM,
        'ian': IAN,
        'ian2m': IAN2m,
        'memnet': MemNet,
        'memnet2': MemNet2,
        'ram': RAM,
        'ram2': MIMN,
        'ram2m': RAM2m,
        'cabasc': Cabasc,
        'mimn': MIMN,
        'end2end_lstm': End2End_LSTM,
        'joint_model': JointModel_v2,
        'multi_joint_model': MultiJointModel_complete_v2,
        'feat_filter_model': FeatFilterModel
    }

    input_colses = {
        'lstm': ['text_raw_indices'],
        'joint_model': ['text_raw_indices', 'aspect_id'],
        'end2end_lstm': ['text_raw_indices', 'aspect_indices', 'aspect_id'],
        'multi_joint_model': ['text_raw_indices', 'aspect_id', 'imgs', 'num_imgs', 'imgs_fine_grain', 'text_raw_indices_trc', 'imgs_scene_level'],
        'feat_filter_model': ['text_raw_indices', 'imgs', 'imgs_fine_grain'],
        'ae_lstm': ['text_raw_indices', 'aspect_indices'],
        'td_lstm': ['text_left_with_aspect_indices', 'text_right_with_aspect_indices'],
        'ian': ['text_raw_indices', 'aspect_indices'],
        'ian2m': ['text_raw_indices', 'aspect_indices', 'imgs', 'num_imgs'],
        'memnet': ['text_raw_indices', 'aspect_indices'],
        'memnet2': ['text_raw_indices', 'aspect_indices', 'imgs', 'num_imgs'],
        'ram': ['text_raw_indices', 'aspect_indices'],
        'mimn': ['text_raw_indices', 'aspect_indices', 'imgs', 'num_imgs'],
        'ram2m': ['text_raw_indices', 'aspect_indices', 'imgs', 'num_imgs'],
        'cabasc': ['text_raw_indices', 'aspect_indices', 'text_left_with_aspect_indices', 'text_right_with_aspect_indices'],
    }
    # using MASAD Dataset, and share emb between word and aspect
    if opt.dataset == 'MASAD' and opt.shared_emb == 1:
        for m in input_colses:
            for i in range(len(input_colses[m])):
                if input_colses[m][i] == 'aspect_indices':
                    input_colses[m][i] = 'aspect_indices_shared'
    
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    
    #opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
    #    if opt.device is None else torch.device(opt.device)
    opt.device = torch.device('cuda:0')
    #opt.device = torch.device('cpu')

    ins = Instructor(opt)
    # ins.test2('zol_cellphone_mimn_best_acc_61.59_f1_60.51_time_2018-09-05-17-11.models')
    # ins.test3('mimn_reproduce/0828/zol_cellphone_mimn_best_acc_58.67_f1_58.08_time_2022-08-28-16-45.models')
    # ins.test_acd('multi_joint_temp3/initial_setting/MASAD_multi_joint_model_best_acc_93.13_f1_93_time_2023-01-05-04-14.models')
    
    if opt.dataset == 'MASAD':
        if opt.expanded == 0:  # Only ASC-task
            ins.run()
        else:  # ACSA-task
            ins.run_acsa()
    else:
        ins.run()
