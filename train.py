# -*- coding: utf-8 -*-

import os
import math
import time
import argparse
import random
import torch
import torch.nn as nn
import numpy as np
from bucket_iterator import BucketIterator
from data_utils import ABSADataReader,build_tokenizer,build_embedding_matrix
from models import OTE

from transformers import BertTokenizer

from torch.cuda.amp import autocast as autocast
from torch.cuda.amp.grad_scaler import GradScaler

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        absa_data_reader = ABSADataReader(data_dir=opt.data_dir,opt=opt)
        if opt.useBert:
            #bert
            tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
            embedding_matrix = torch.tensor([0])
        else:
            tokenizer = build_tokenizer(data_dir=opt.data_dir)
            embedding_matrix = build_embedding_matrix(opt.data_dir, tokenizer.word2idx, opt.embed_dim, opt.dataset)

        self.idx2tag, self.idx2polarity,self.idx2target ,self.idx2express= absa_data_reader.reverse_tag_map, \
                                          absa_data_reader.reverse_polarity_map,\
                                          absa_data_reader.reverse_target_map,absa_data_reader.reverse_express_map

        self.train_data_loader = BucketIterator(data=absa_data_reader.get_train(tokenizer),
                                                batch_size=opt.batch_size,
                                                shuffle=True)
        self.dev_data_loader = BucketIterator(data=absa_data_reader.get_dev(tokenizer),
                                              batch_size=opt.batch_size,
                                              shuffle=False)
        self.model = opt.model_class(embedding_matrix=embedding_matrix,
                                     opt=opt,
                                     idx2tag=self.idx2tag,
                                     idx2polarity=self.idx2polarity,
                                     idx2target = self.idx2target,
                                     idx2express=self.idx2express
                                     ).to(opt.device)
        self._print_args()

        if torch.cuda.is_available():
            print('>>> cuda memory allocated:', torch.cuda.memory_allocated(device=opt.device.index))

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('>>> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('>>> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    torch.nn.init.xavier_uniform_(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv) #均匀分布 U(a,b)

    def _train(self):
        if not os.path.exists('state_dict/'):
            os.mkdir('state_dict/')

        self._reset_params()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())

        optimizer = torch.optim.Adam(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
        #自动调整学习率
        decay, decay_step = self.opt.decay, self.opt.decay_steps
        l = lambda epoch: decay ** (epoch // decay_step)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=l)

        scaler = GradScaler()#自动混合精度
        print("**************  Start trainging **************")
        max_dev_f1 = 0.0
        best_state_dict_path = ''
        global_step = 0
        continue_not_increase = 0
        for epoch in range(self.opt.num_epoch):
            print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
            print('>' * 100)
            print('epoch: {0}'.format(epoch+1))
            increase_flag = False
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                start_time = time.time()
                global_step += 1
                scheduler.step()
                # switch model to training mode, clear gradient accumulators
                self.model.train()

                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.input_cols]
                targets = [sample_batched[col].to(self.opt.device) for col in self.opt.target_cols]
                with autocast():
                    outputs = self.model(inputs,opt)#模型只喂入了text
                    loss = self.model.calc_loss(outputs, targets)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()


                optimizer.zero_grad()
                end_time = time.time()
                print("epoch:%d, batch:%d,time:%.2f, loss:%.3f" % (epoch+1,i_batch,end_time-start_time,loss.item()))
                if global_step % self.opt.log_step == 0:

                    nn.utils.clip_grad_norm_(_params,max_norm=opt.clip)
                    dev_ap_metrics, dev_op_metrics, dev_triplet_metrics, dev_senPolarity_metrics,dev_target_metrics,dev_express_metrics = self._evaluate(self.dev_data_loader)

                    dev_ap_precision, dev_ap_recall, dev_ap_f1 = dev_ap_metrics
                    dev_op_precision, dev_op_recall, dev_op_f1 = dev_op_metrics
                    dev_triplet_precision, dev_triplet_recall, dev_triplet_f1 = dev_triplet_metrics
                    dev_target_precision, dev_target_recall, dev_target_f1 = dev_target_metrics
                    dev_express_precision, dev_express_recall, dev_express_f1 = dev_express_metrics
                    # dev_RE_precision, dev_RE_recall, dev_RE_f1 = dev_RE_metrics
                    print('dev_ap_precision: {:.4f}, dev_ap_recall: {:.4f}, dev_ap_f1: {:.4f}'.format(dev_ap_precision, dev_ap_recall, dev_ap_f1))
                    print('dev_op_precision: {:.4f}, dev_op_recall: {:.4f}, dev_op_f1: {:.4f}'.format(dev_op_precision, dev_op_recall, dev_op_f1))
                    print('dev_triplet_precision: {:.4f}, dev_triplet_recall: {:.4f}, dev_triplet_f1: {:.4f}'.format( dev_triplet_precision, dev_triplet_recall, dev_triplet_f1))
                    print('dev_target_precision: {:.4f}, dev_target_recall: {:.4f}, dev_target_f1: {:.4f}'.format( dev_target_precision, dev_target_recall, dev_target_f1))
                    print('dev_express_precision: {:.4f}, dev_express_recall: {:.4f}, dev_express_f1: {:.4f}'.format( dev_express_precision, dev_express_recall, dev_express_f1))
                    print('dev_senPolarity_acc: {:.4f}'.format(dev_senPolarity_metrics))
                    if dev_triplet_f1 > max_dev_f1:
                        increase_flag = True
                        print("history:%.4f, current:%.4f " % (max_dev_f1, dev_triplet_f1))
                        max_dev_f1 = dev_triplet_f1
                        if self.opt.useBert:
                            best_state_dict_path = 'state_dict/'+'ote_Bert'+'_'+self.opt.dataset+'.pkl'
                        else:
                            best_state_dict_path = 'state_dict/'+'ote_LSTM'+'_'+self.opt.dataset+'.pkl'
                        torch.save(self.model.state_dict(), best_state_dict_path)
                        print('>>> best model saved.')

        print("************** Finish train **************")

        return best_state_dict_path

    def _evaluate(self, data_loader):
        # switch model to evaluation mode
        print("********** Strat eval **********")
        self.model.eval()

        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.input_cols]
                t_ap_spans, t_op_spans, t_triplets, t_senPolarity ,t_target,t_express= [t_sample_batched[col] for col in self.opt.eval_cols]
                t_senPolarity = t_senPolarity.cpu().numpy().tolist()

                start_time = time.time()
                with autocast():
                    dev_outpus = self.model(t_inputs,opt)
                model_time = time.time()-start_time
                t_ap_spans_pred, t_op_spans_pred, t_triplets_pred, t_senPolarity_pred,t_target_pred ,t_express_pred= self.model.inference(dev_outpus,t_inputs[0],t_inputs[1])
                t_senPolarity_pred = t_senPolarity_pred.cpu().numpy().tolist()
                infer_time = time.time()-start_time
                print("model_time:%.2f, infer_time:%.2f " %(model_time,infer_time))


        return self._metrics(t_ap_spans, t_ap_spans_pred), \
               self._metrics(t_op_spans, t_op_spans_pred), \
               self._metrics(t_triplets, t_triplets_pred), \
               self._metrics_senPolarity(t_senPolarity,t_senPolarity_pred), \
               self._metrics(t_target, t_target_pred),\
               self._metrics(t_express,t_express_pred)

    @staticmethod
    def _metrics(targets, outputs):
        TP, FP, FN = 0, 0, 0
        n_sample = len(targets)
        assert n_sample == len(outputs)
        for i in range(n_sample):
            n_hit = 0
            n_output = len(outputs[i])
            n_target = len(targets[i])
            for t in outputs[i]:
                if t in targets[i]:
                    n_hit += 1
            TP += n_hit
            FP += (n_output - n_hit)
            FN += (n_target - n_hit)
        precision = float(TP) / float(TP + FP + 1e-5)
        recall = float(TP) / float(TP + FN + 1e-5)
        f1 = 2 * precision * recall / (precision + recall + 1e-5)
        return [precision, recall, f1]

    @staticmethod
    def _metrics_senPolarity(targets, outputs):
        targets = np.array(targets)
        outputs = np.array(outputs)
        correct = np.sum(targets==outputs)
        acc = correct / len(targets)
        return acc



    # def run(self):
    #     if not os.path.exists('state_dict/'):
    #         os.mkdir('state_dict/')
    #
    #     # self._reset_params()
    #     _params = filter(lambda p: p.requires_grad, self.model.parameters())
    #
    #     optimizer = torch.optim.Adam(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
    #     #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=l)
    #
    #     self._train(optimizer)



if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ote', type=str)
    parser.add_argument('--useBert', default=False, type=bool,help='if Fasle , will ues LSTM')
    parser.add_argument('--dataset', default='hotel', type=str, help='hotel,test')
    parser.add_argument('--learning_rate', default=0.00001, type=float)
    parser.add_argument('--l2reg', default=0.00001, type=float)
    parser.add_argument('--num_epoch', default=1000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--log_step', default=500, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--polarities_dim', default=4, type=int)
    parser.add_argument('--seed', default=776, type=int)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--clip', default=5.0, type=float)
    parser.add_argument('--decay', default=0.75, type=float)
    parser.add_argument('--decay_steps', default=500, type=float)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    opt = parser.parse_args()

    model_classes = {
        'ote': OTE,
    }
    input_colses = {
        'ote': ['text_indices', 'text_mask'],
    }
    target_colses = {
        'ote': ['ap_indices', 'op_indices', 'triplet_indices', 'text_mask','sentece_polarity','target_indices','express_indices'],
    }

    data_dirs = {
        'hotel' : 'hotelDatasets/hotel',
        'test'  : 'hotelDatasets/test'
    }
    opt.model_class = model_classes[opt.model]
    opt.input_cols = input_colses[opt.model]
    opt.target_cols = target_colses[opt.model]
    opt.eval_cols = ['ap_spans', 'op_spans', 'triplets','sentece_polarity','targets','express_label']
    opt.data_dir = data_dirs[opt.dataset]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ins = Instructor(opt)
    # ins.run()
    ins._train()