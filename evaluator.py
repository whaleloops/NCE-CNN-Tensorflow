import numpy as np
import random
import sys

class SentencePairEvaluator:
    def __init__(self, model, sess, dataset):
        self.model = model
        self.sess = sess
        self.dataset = dataset
        self.max_length = model._seq_length
        
    def padding(self, sent):
        length = len(sent)
        sent_ = sent[:]
        if length < self.max_length:
            sent_.extend([0] * (self.max_length - length))
        else:
            sent_ = sent_[:self.max_length]
        return sent_
   
    def _map(self, scores, labels):
        score_rankings = scores.argsort()[::-1]
        tp, ap = 0, 0
        for i in range(len(scores)):
            if labels[score_rankings[i]] == 1:
                tp = tp + 1.0
                ap = ap + tp / (i+1.0)
        return ap / (labels.sum() + 1e-8)
         
    def _mrr(self, scores, labels):
        score_rankings = scores.argsort()[::-1]
        mrr_score = 0
        for i in range(len(scores)):
            if labels[score_rankings[i]] == 1:
                mrr_score = 1.0 / (i + 1)
                break
        return mrr_score
        
    def evaluate(self):
        maps = []
        mrrs = []
        #print self.dataset['boundary']
        for i in range(len(self.dataset['boundary']) - 1):
            left_bound, right_bound = self.dataset['boundary'][i], self.dataset['boundary'][i+1]
            #print i, left_bound, right_bound, len(self.dataset['lsents'])
            questions = np.array([self.padding(self.dataset['lsents'][sid]) for sid in range(left_bound, right_bound)], dtype=np.int32)
            #print questions.shape
            #print questions
            q_lens = np.array([len(self.dataset['lsents'][sid]) for sid in range(left_bound, right_bound)], dtype=np.int32)
            answers = np.array([self.padding(self.dataset['rsents'][sid]) for sid in range(left_bound, right_bound)], dtype=np.int32)
            #print answers.shape
            #print answers
            a_lens = np.array([len(self.dataset['rsents'][sid]) for sid in range(left_bound, right_bound)], dtype=np.int32)
            labels = np.array([self.dataset['labels'][sid] for sid in range(left_bound, right_bound)], dtype=np.int32)
            #print labels
            feed_dict = {self.model.input_questions: questions,
                         self.model.input_question_lens: q_lens,
                         self.model.input_answers: answers,
                         self.model.input_answer_lens: a_lens,
                         self.model.keep_prob: np.float32(1.0)}
            scores = self.sess.run(self.model.scores, feed_dict=feed_dict)
            #print scores
            maps.append(self._map(scores, labels))
            mrrs.append(self._mrr(scores, labels))
        return np.mean(maps), np.mean(mrrs)
