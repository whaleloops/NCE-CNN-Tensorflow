import sys
import time
import numpy as np
import tensorflow as tf
from collections import defaultdict
from random import shuffle
import pdb

from evaluator import *

class DataGeneratePointWise:
    def __init__(self, model, sess, data, batch_size, max_length, sampling = 'max'):
        self.model = model
        self.data = data
        self.data_size = data['size']
        self.batch_size = batch_size
        self.max_length = max_length
    
    def padding(self, sent):
        length = len(sent)
        sent_ = sent[:]
        if length < self.max_length:
            sent_.extend([0] * (self.max_length - length))
        else:
            sent_ = sent_[:self.max_length]
        return sent_
    
    def next(self):
        num_classes = 2
        samples = np.random.randint(self.data_size, size=self.batch_size)
        qs, qlens = [], []
        ans, alens = [], []
        labels_arr = []
        for id in samples:
            bound, posbound, negbound = self.data['id2boundary'][id]
            id_ = np.random.randint(bound[0], bound[1])
            if id_ < posbound[1]:
                label = 1
            else:
                label = 0
            qs.append(self.padding(self.data['lsents'][id_]))
            qlens.append(len(self.data['lsents'][id_]))
            ans.append(self.padding(self.data['rsents'][id_]))
            alens.append(len(self.data['rsents'][id_]))
            labels_arr.append(label)
        # To one hot
        labels_arr = np.array(labels_arr)
        labels = np.zeros((labels_arr.shape[0], num_classes))
        labels[np.arange(labels_arr.shape[0]), labels_arr] = 1
        feed_dict = {self.model.input_questions: np.array(qs, dtype = np.int32), 
                     self.model.input_question_lens: np.array(qlens, dtype = np.int32),
                     self.model.input_answers: np.array(ans, dtype = np.int32), 
                     self.model.input_answer_lens: np.array(alens, dtype = np.int32),
                     self.model.labels: np.array(labels, dtype = np.int32),
                     self.model.keep_prob: np.float32(0.5)}
        return feed_dict

class DataGeneratePairWise:
    def __init__(self, model, sess, data, batch_size, max_length, sampling='max', sample_num=5):
        self.model = model
        self.data = data
        self.data_size = data['size']
        self.batch_size = batch_size
        self.max_length = max_length
        self.sampling = sampling
        self.sess = sess
    
    def padding(self, sent):
        length = len(sent)
        sent_ = sent[:]
        if length < self.max_length:
            sent_.extend([0] * (self.max_length - length))
        else:
            sent_ = sent_[:self.max_length]
        return sent_
    
    def next(self):
        pos_qs, pos_qlens = [], []
        pos_as, pos_alens = [], []
        neg_as, neg_alens = [], []
        for i in range(self.batch_size):
            while 1:
                sample_id = np.random.randint(self.data_size)
                bound, posbound, negbound = self.data['id2boundary'][sample_id]
                if posbound[0] < posbound[1] and negbound[0] < negbound[1]:
                    break
            # random
            if self.sampling != 'max':
                posid = np.random.randint(posbound[0], posbound[1])
                negid = np.random.randint(negbound[0], negbound[1])
            else:
            # min-max
                questions = np.array([self.padding(self.data['lsents'][sid]) for sid in range(negbound[0], bound[1])], dtype=np.int32)
                q_lens = np.array([len(self.data['lsents'][sid]) for sid in range(negbound[0], bound[1])], dtype=np.int32)
                answers = np.array([self.padding(self.data['rsents'][sid]) for sid in range(negbound[0], bound[1])], dtype=np.int32)
                a_lens = np.array([len(self.data['rsents'][sid]) for sid in range(negbound[0], bound[1])], dtype=np.int32)
                feed_dict = {self.model.input_questions: questions,
                         self.model.input_question_lens: q_lens,
                         self.model.input_answers: answers,
                         self.model.input_answer_lens: a_lens,
                         self.model.keep_prob: np.float32(1.0)}
                scores = self.sess.run(self.model.scores, feed_dict=feed_dict)
                posid = np.random.randint(posbound[0], posbound[1])
                negid = np.argmax(scores[:]) + negbound[0]
                #posid = np.argmin(scores[:posbound[1]-posbound[0]]) + posbound[0]
                #negid = np.argmax(scores[posbound[1]-posbound[0]:]) + negbound[0]

            
            pos_qs.append(self.padding(self.data['lsents'][posid]))
            pos_qlens.append(len(self.data['lsents'][posid]))
            #print len(self.data['lsents'][posid])
            pos_as.append(self.padding(self.data['rsents'][posid]))
            pos_alens.append(len(self.data['rsents'][posid]))
            neg_as.append(self.padding(self.data['rsents'][negid]))
            neg_alens.append(len(self.data['rsents'][negid]))
        feed_dict = {self.model.input_questions: np.array(pos_qs, dtype = np.int32), 
                     self.model.input_question_lens: np.array(pos_qlens, dtype = np.int32),
                     self.model.input_answers: np.array(pos_as, dtype = np.int32), 
                     self.model.input_answer_lens: np.array(pos_alens, dtype = np.int32),
                     self.model.neg_answers: np.array(neg_as, dtype = np.int32), 
                     self.model.neg_answer_lens: np.array(neg_alens, dtype = np.int32),
                     self.model.keep_prob: np.float32(0.5)}
        return feed_dict
