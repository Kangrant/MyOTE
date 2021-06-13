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
from data_utils import ABSADataReader
from models import CMLA, HAST, OTE

from transformers import BertTokenizer

from torch.cuda.amp import autocast as autocast
from torch.cuda.amp.grad_scaler import GradScaler

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        

        absa_data_reader = ABSADataReader(data_dir=opt.data_dir)

        #bert
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

        self.idx2tag, self.idx2polarity,self.idx2target ,self.idx2express= absa_data_reader.reverse_tag_map, \
                                          absa_data_reader.reverse_polarity_map,\
                                          absa_data_reader.reverse_target_map,absa_data_reader.reverse_express_map

        self.train_data_loader = BucketIterator(data=absa_data_reader.get_train(tokenizer),
                                                batch_size=opt.batch_size,
                                                shuffle=True)
        self.dev_data_loader = BucketIterator(data=absa_data_reader.get_dev(tokenizer),
                                              batch_size=opt.batch_size,
                                              shuffle=False)

        self.model = opt.model_class(
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


    def _train(self, optimizer):
        # if os.path.exists("./temp/model.bin"):
        #     torch.load()
        scaler = GradScaler()#自动混合精度
        print("**************  Start trainging **************")
        max_dev_f1 = 0.0
        best_state_dict_path = ''
        global_step = 0
        continue_not_increase = 0
        for epoch in range(self.opt.num_epoch):

            print('>' * 100)
            print('epoch: {0}'.format(epoch+1))
            increase_flag = False
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                start_time = time.time()
                global_step += 1

                # switch model to training mode, clear gradient accumulators
                self.model.train()

                #optimizer.zero_grad()

                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.input_cols]
                targets = [sample_batched[col].to(self.opt.device) for col in self.opt.target_cols]
                with autocast():
                    outputs = self.model(inputs)#模型只喂入了text
                    loss = self.model.calc_loss(outputs, targets)
                #loss.backward()
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()


                optimizer.zero_grad()
                end_time = time.time()
                #optimizer.step()
                print("epoch:%d, batch:%d,time:%.2f, loss:%.3f" % (epoch+1,i_batch,end_time-start_time,loss.item()))
                if global_step % self.opt.log_step == 0:


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
                        print("history:%.4f, current:%.4f "%(max_dev_f1,dev_triplet_f1))
                        max_dev_f1 = dev_triplet_f1
                        best_state_dict_path = 'state_dict/'+self.opt.model+'_'+self.opt.dataset+'.pkl'
                        torch.save(self.model.state_dict(), best_state_dict_path)
                        print('>>> best model saved.')

            if increase_flag == False:
                continue_not_increase += 1
                if continue_not_increase >= self.opt.patience:
                    print('early stop.')
                    # break
            else:
                continue_not_increase = 0
        print("************** Finish train **************")
        # torch.save(self.model.state_dict(),'./temp/model.bin')
        # print(" saving model successful")
        return best_state_dict_path

    def _evaluate(self, data_loader):
        # switch model to evaluation mode
        print("********** Strat eval **********")
        self.model.eval()
        t_ap_spans_all, t_op_spans_all, t_triplets_all  = None, None, None
        t_ap_spans_pred_all, t_op_spans_pred_all, t_triplets_pred_all  = None, None, None

        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.input_cols]
                t_ap_spans, t_op_spans, t_triplets, t_senPolarity ,t_target,t_express= [t_sample_batched[col] for col in self.opt.eval_cols]
                t_senPolarity = t_senPolarity.cpu().numpy().tolist()
                start_time = time.time()
                with autocast():
                    dev_outpus = self.model(t_inputs)
                model_time = time.time()-start_time
                #t_ap_spans_pred, t_op_spans_pred, t_triplets_pred = self.model.inference(t_inputs)
                t_ap_spans_pred, t_op_spans_pred, t_triplets_pred, t_senPolarity_pred,t_target_pred ,t_express_pred= self.model.inference(dev_outpus,t_inputs[0],t_inputs[1])
                t_senPolarity_pred = t_senPolarity_pred.cpu().numpy().tolist()
                infer_time = time.time()-start_time

                print("model_time:%.2f, infer_time:%.2f " %(model_time,infer_time))

                if t_ap_spans_all is None:
                    t_ap_spans_all = t_ap_spans
                    t_op_spans_all = t_op_spans
                    t_triplets_all = t_triplets
                    t_target_all = t_target
                    t_senPolarity_all = t_senPolarity
                    t_express_all = t_express

                    t_ap_spans_pred_all = t_ap_spans_pred
                    t_op_spans_pred_all = t_op_spans_pred
                    t_triplets_pred_all = t_triplets_pred
                    t_target_pred_all = t_target_pred
                    t_senPolarity_pred_all = t_senPolarity_pred
                    t_express_pred_all = t_express_pred
                else:
                    t_ap_spans_all = t_ap_spans_all + t_ap_spans
                    t_op_spans_all = t_op_spans_all + t_op_spans
                    t_triplets_all = t_triplets_all + t_triplets
                    t_target_all = t_target_all + t_target
                    t_senPolarity_all = t_senPolarity_all + t_senPolarity
                    t_express_all = t_express_all + t_express

                    t_ap_spans_pred_all = t_ap_spans_pred_all + t_ap_spans_pred
                    t_op_spans_pred_all = t_op_spans_pred_all + t_op_spans_pred
                    t_triplets_pred_all = t_triplets_pred_all + t_triplets_pred
                    t_target_pred_all = t_target_pred_all + t_target_pred
                    t_senPolarity_pred_all = t_senPolarity_pred_all + t_senPolarity_pred
                    t_express_pred_all = t_express_pred_all + t_express_pred

        return self._metrics(t_ap_spans_all, t_ap_spans_pred_all), \
               self._metrics(t_op_spans_all, t_op_spans_pred_all), \
               self._metrics(t_triplets_all, t_triplets_pred_all), \
               self._metrics_senPolarity(t_senPolarity_all,t_senPolarity_pred_all), \
               self._metrics(t_target_all, t_target_pred_all),\
               self._metrics(t_express_all,t_express_pred_all)

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



    def run(self, repeats=1):
        if not os.path.exists('log/'):
            os.mkdir('log/')

        if not os.path.exists('state_dict/'):
            os.mkdir('state_dict/')
        if self.opt.v2:
            f_out = open('log/'+self.opt.model+'_'+self.opt.dataset+'_val_v2.txt', 'w', encoding='utf-8')

        else:
            f_out = open('log/'+self.opt.model+'_'+self.opt.dataset+'_val.txt', 'w', encoding='utf-8')

        test_ap_precision_avg = 0
        test_ap_recall_avg = 0
        test_ap_f1_avg = 0

        test_op_precision_avg = 0
        test_op_recall_avg = 0
        test_op_f1_avg = 0

        test_triplet_precision_avg = 0
        test_triplet_recall_avg = 0
        test_triplet_f1_avg = 0

        for i in range(repeats):
            print('repeat: {0}'.format(i+1))
            f_out.write('repeat: {0}\n'.format(i+1))

            # self._reset_params()
            _params = filter(lambda p: p.requires_grad, self.model.parameters())

            optimizer = torch.optim.Adam(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
            #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=l)

            best_state_dict_path = self._train(optimizer)
            print("************** Start testing **************")

            self.model.load_state_dict(torch.load(best_state_dict_path))

            print("  Load model successful  ")

            test_ap_metrics, test_op_metrics, test_triplet_metrics = self._evaluate(self.dev_data_loader)
            test_ap_precision, test_ap_recall, test_ap_f1 = test_ap_metrics
            test_op_precision, test_op_recall, test_op_f1 = test_op_metrics
            test_triplet_precision, test_triplet_recall, test_triplet_f1 = test_triplet_metrics

            print('test_ap_precision: {:.4f}, test_ap_recall: {:.4f}, test_ap_f1: {:.4f}'
                  .format(test_ap_precision, test_ap_recall, test_ap_f1))
            f_out.write('test_ap_precision: {:.4f}, test_ap_recall: {:.4f}, test_ap_f1: {:.4f}\n'
                        .format(test_ap_precision, test_ap_recall, test_ap_f1))
            print('test_op_precision: {:.4f}, test_op_recall: {:.4f}, test_op_f1: {:.4f}'
                  .format(test_op_precision, test_op_recall, test_op_f1))
            f_out.write('test_op_precision: {:.4f}, test_op_recall: {:.4f}, test_op_f1: {:.4f}\n'
                        .format(test_op_precision, test_op_recall, test_op_f1))
            print('test_triplet_precision: {:.4f}, test_triplet_recall: {:.4f}, test_triplet_f1: {:.4f}'
                  .format(test_triplet_precision, test_triplet_recall, test_triplet_f1))
            f_out.write('test_triplet_precision: {:.4f}, test_triplet_recall: {:.4f}, test_triplet_f1: {:.4f}\n'
                        .format(test_triplet_precision, test_triplet_recall, test_triplet_f1))

            test_ap_precision_avg += test_ap_precision
            test_ap_recall_avg += test_ap_recall
            test_ap_f1_avg += test_ap_f1

            test_op_precision_avg += test_op_precision
            test_op_recall_avg += test_op_recall
            test_op_f1_avg += test_op_f1

            test_triplet_precision_avg += test_triplet_precision
            test_triplet_recall_avg += test_triplet_recall
            test_triplet_f1_avg += test_triplet_f1

            print('#' * 100)

        print("test_ap_precision_avg:", test_ap_precision_avg / repeats)
        print("test_ap_recall_avg:", test_ap_recall_avg / repeats)
        print("test_ap_f1_avg:", test_ap_f1_avg / repeats)
        print("test_op_precision_avg:", test_op_precision_avg / repeats)
        print("test_op_recall_avg:", test_op_recall_avg / repeats)
        print("test_op_f1_avg:", test_op_f1_avg / repeats)
        print("test_triplet_precision_avg:", test_triplet_precision_avg / repeats)
        print("test_triplet_recall_avg:", test_triplet_recall_avg / repeats)
        print("test_triplet_f1_avg:", test_triplet_f1_avg / repeats)

        f_out.close()
        print("************** Test Finished *************")


if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--v2', action='store_true')
    parser.add_argument('--model', default='ote', type=str)
    parser.add_argument('--dataset', default='hotel', type=str, help='laptop14, rest14, rest15, rest16')
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
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
    opt = parser.parse_args()

    model_classes = {
        'cmla': CMLA,
        'hast': HAST,
        'ote': OTE,
    }
    input_colses = {
        'cmla': ['text_indices', 'text_mask'],
        'hast': ['text_indices', 'text_mask'],
        'ote': ['text_indices', 'text_mask'],
    }
    target_colses = {
        'cmla': ['ap_indices', 'op_indices', 'triplet_indices', 'text_mask'],
        'hast': ['ap_indices', 'op_indices', 'triplet_indices', 'text_mask'],
        'ote': ['ap_indices', 'op_indices', 'triplet_indices', 'text_mask','sentece_polarity','target_indices','express_indices'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    data_dirs = {
        'laptop14': 'datasets/14lap',
        'rest14': 'datasets/14rest',
        'rest15': 'datasets/15rest',
        'rest16': 'datasets/16rest',
        'hotel' : 'hotelDatasets/hotel',
        'test'  : 'hotelDatasets/test'
    }
    opt.model_class = model_classes[opt.model]
    opt.input_cols = input_colses[opt.model]
    opt.target_cols = target_colses[opt.model]
    opt.eval_cols = ['ap_spans', 'op_spans', 'triplets','sentece_polarity','targets','express_label']
    opt.initializer = initializers[opt.initializer]
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
    ins.run()
