# On-going construction
# NON-debugged code

import argparse
import sys, os
import numpy as np

from util.Vocab import Vocab
from util.read_data import read_relatedness_dataset, read_embedding

from model import *
from evaluator import *
from data_generate import * 
FLAGS = tf.app.flags.FLAGS

def load_glove_word_embeddings(vocab):
    emb_dir = 'data/glove/'
    emb_prefix = emb_dir + 'glove.840B'
    emb_vocab, emb_vecs = read_embedding(emb_prefix + '.vocab', emb_prefix + '.300d.npy')
    emb_dim = emb_vecs.shape[1]

    # use only vectors in vocabulary (not necessary, but gives faster training)

    num_unk = 0
    vecs = np.zeros([vocab.size, emb_dim])

    UNK = np.random.uniform(low=-0.05, high=0.05, size=emb_dim)

    for i in range(0, vocab.size):
        w = vocab.token(i)
        if emb_vocab.contains(w):
            vecs[i] = emb_vecs[emb_vocab.index(w)]
        else:
            vecs[i] = emb_vecs[emb_vocab.index('unk')] #UNK --:uniform(-0.05, 0.05)
            num_unk = num_unk + 1
    print('unk count = %d' % num_unk)
    return vecs

def train(train_dataset, dev_dataset, test_dataset, vecs, iter_num = 10000):
    if FLAGS.model == 'average-pointwise':
        model_name = SentencePairEncoder
        evaluator_name = SentencePairEvaluator
        data_generator_name = DataGeneratePointWise
    elif FLAGS.model == 'average-pairwise':
        model_name = SentencePairEncoderPairwiseRanking
        evaluator_name = SentencePairEvaluator
        data_generator_name = DataGeneratePairWise
    elif FLAGS.model == 'cnn-pointwise':
        model_name = SentencePairEncoderCNN
        evaluator_name = SentencePairEvaluator
        data_generator_name = DataGeneratePointWise
    elif FLAGS.model == 'mpssn-pointwise':
        model_name = SentencePairEncoderMPSSN
        evaluator_name = SentencePairEvaluator
        data_generator_name = DataGeneratePointWise
    elif FLAGS.model == 'cnn-pairwise':
        model_name = SentencePairEncoderPairwiseRankingCNN
        evaluator_name = SentencePairEvaluator
        data_generator_name = DataGeneratePairWise
    elif FLAGS.model == 'rnn-pairwise':
        model_name = SentencePairEncoderPairwiseRankingGRU
        evaluator_name = SentencePairEvaluator
        data_generator_name = DataGeneratePairWise
        
        
    model = model_name(vecs, dim=FLAGS.dim, seq_length=FLAGS.max_length, num_filters=[300,20]) #TODO change num_filters to [300,20]
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)
    saver = tf.train.Saver()
    #train = optimizer.minimize(model.loss)
    tvars = tf.trainable_variables() 
    grads, _ = tf.clip_by_global_norm(tf.gradients(model.loss, tvars), 10)
    train = optimizer.apply_gradients(zip(grads, tvars), global_step=model.global_step)
    
    print('Start training')
    tf.set_random_seed(1234)
    np.random.seed(1234)
    best_dev_map, best_dev_mrr = 0, 0
    best_test_map, best_test_mrr = 0, 0
    best_model = None
    best_iter = 0
    not_improving = 0
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        data_generator = data_generator_name(model, sess, train_dataset, FLAGS.batch_size, FLAGS.max_length, FLAGS.sampling)
        dev_evaluator = evaluator_name(model, sess, dev_dataset)
        test_evaluator = evaluator_name(model, sess, test_dataset)
        if FLAGS.load_model:
            saver.restore(sess, FLAGS.load_model)
        
        for iter in range(iter_num):
            feed_dict = data_generator.next()
            #print feed_dict 
            _, loss = sess.run([train, model.loss], feed_dict=feed_dict)
            if iter%2 == 0: # TODO: change 2 to 50
                print('%d iter, loss = %.5f' %(iter, loss))
            if iter%FLAGS.eval_freq == 0:
                dev_map, dev_mrr = dev_evaluator.evaluate()
                test_map, test_mrr = test_evaluator.evaluate()
                if dev_map > best_dev_map:
                    not_improving = 0
                    best_dev_map, best_dev_mrr = dev_map, dev_mrr
                    best_test_map, best_test_mrr = test_map, test_mrr
                    best_iter = iter
                    print('New best valid MAP!')
                    if FLAGS.save_path:
                        if not os.path.isdir(FLAGS.save_path):
                            os.mkdir(FLAGS.save_path)
                        save_path = saver.save(sess, FLAGS.save_path + '/model.tf')
                        print("Model saved")
                else:
                    not_improving += 1
                    if not_improving > 3:
                        break
                print('%d iter, dev: MAP %.3f  MRR %.3f' %(iter, dev_map, dev_mrr))
                print('%d iter, test:  MAP %.3f  MRR %.3f' %(iter, test_map, test_mrr))
                print('Best at iter %d, valid %.3f, test %.3f\n' %(best_iter, best_dev_map, best_test_map))
                if not_improving > 3:
                    break
        print('\n\nFinish training!')
        print('Performance dev: MAP %.3f   MRR %.3f <==' %(best_dev_map, best_dev_mrr))
        print('Performance test: MAP %.3f   MRR %.3f' %(best_test_map, best_test_mrr))

def main(argv):
    if FLAGS.dataset != 'TrecQA' and FLAGS.dataset != 'WikiQA':
        print('Error dataset!')
        sys.exit()

    # directory containing dataset files
    data_dir = 'data/' + FLAGS.dataset + '/'

    # load vocab
    vocab = Vocab(data_dir + 'vocab.txt')

    # load embeddings
    print('loading glove word embeddings')
    vecs = load_glove_word_embeddings(vocab)

    # load datasets
    print('loading datasets' + FLAGS.dataset)
    if FLAGS.dataset == 'TrecQA':
        train_dir = data_dir + 'train-all/'
        dev_dir = data_dir + FLAGS.version + '-dev/'
        test_dir = data_dir + FLAGS.version + '-test/'
    elif FLAGS.dataset == 'WikiQA':
        train_dir = data_dir + 'train/'
        dev_dir = data_dir + 'dev/'
        test_dir = data_dir + 'test/'

    train_dataset = read_relatedness_dataset(train_dir, vocab) # This is a dict
    dev_dataset = read_relatedness_dataset(dev_dir, vocab)
    test_dataset = read_relatedness_dataset(test_dir, vocab)
    print('train_dir: %s, num train = %d' % (train_dir, train_dataset['size']))
    print('dev_dir: %s, num dev = %d' % (dev_dir, dev_dataset['size']))
    print('test_dir: %s, num test = %d' % (test_dir, test_dataset['size']))

    train(train_dataset, dev_dataset, test_dataset, vecs)

if __name__ == '__main__':
    tf.app.flags.DEFINE_string('dataset', 'TrecQA', 'dataset, can be TrecQA or WikiQA')
    tf.app.flags.DEFINE_string('version', 'raw', 'the version of TrecQA dataset, can be raw and clean')
    tf.app.flags.DEFINE_string('model', 'mpssn-pointwise', 'the version of model to be used')
    #tf.app.flags.DEFINE_string('num_pairs', 8, 'number of negative samples for each pos sample')
    tf.app.flags.DEFINE_string('dim', 150, 'dimension of hidden layers')
    tf.app.flags.DEFINE_string('lr', 1e-3, 'learning rate')
    tf.app.flags.DEFINE_string('batch_size', 2, 'mini-batch size') #TODO: change batch size to 64
    tf.app.flags.DEFINE_string('max_length', 48, 'max sentence length')
    tf.app.flags.DEFINE_string('eval_freq', 1000, 'evaluate every x batches')
    tf.app.flags.DEFINE_string('sampling', 'random', 'sampling strategy, max or random')
    tf.app.flags.DEFINE_string('load_model', '', 'path to load model')
    tf.app.flags.DEFINE_string('save_path', 'save_models', 'path to save model')
    
    tf.app.run()
