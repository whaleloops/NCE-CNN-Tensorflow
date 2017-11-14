# On-going construction
# NON-debugged code

import argparse
import sys, os, socket
import numpy as np

from util.Vocab import Vocab
from util.read_data import read_relatedness_dataset, read_embedding

from model import *
from evaluator import *
from data_generate import * 

FLAGS = tf.app.flags.FLAGS
tf.logging.set_verbosity(tf.logging.WARN)

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
    print '----MODEL TRAINING----'
    print 'Start building models'
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
        
        
    model = model_name(vecs, dim=FLAGS.dim, seq_length=FLAGS.max_length, regularization = FLAGS.reg, num_filters=[FLAGS.filterA,FLAGS.filterB])
    #check_op = tf.add_check_numerics_ops()
    if FLAGS.update_tech=='adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)
    else:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.lr)
        
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
        data_generator = data_generator_name(model, sess, train_dataset, FLAGS.batch_size, FLAGS.max_length, FLAGS.sampling, FLAGS.keep_prob)
        dev_evaluator = evaluator_name(model, sess, dev_dataset)
        test_evaluator = evaluator_name(model, sess, test_dataset)
        if FLAGS.load_model:
            saver.restore(sess, FLAGS.load_model)
        
        for iter in range(iter_num):
            feed_dict = data_generator.next()
            #print feed_dict
            #_, loss, score, out, fea_h, fea_a, fea_b, embed_layer, embed_layer_mask, embeddings, W1, b1, W2, b2, Wh, bh, Wo  = sess.run([train, model.loss, model.scores, model.out, model.fea_h, model.fea_a, model.fea_b, model.embed_layer, model.embed_layer_mask, model.word_embeddings, model.W1, model.b1, model.W2, model.b2, model.Wh, model.bh, model.Wo], feed_dict=feed_dict)
            _, loss = sess.run([train, model.loss], feed_dict=feed_dict)
            '''
            print 'the:'
            print embeddings[52270]
            print 'embed_layer:' 
            print embed_layer.shape
            print embed_layer[0]
            print 'embed_layer_mask:' 
            print embed_layer_mask.shape
            print embed_layer_mask[0]
            print 'fea_h:' 
            print len(fea_h) 
            print fea_h[0].shape
            print fea_h
            print 'fea_a:'
            print len(fea_a) 
            print fea_a[0].shape
            print fea_a
            print 'fea_b:'
            print len(fea_b) 
            print fea_b[0].shape
            print fea_b
            print 'out:'
            print out.shape
            print out[0]
            print 'score:'
            print score.shape
            print score[0]
            print 'W1_0'
            print W1[0].shape
            print W1[0]
            print 'W1_1'
            print W1[1].shape
            print W1[1]
            print 'W1_2'
            print W1[2].shape
            print W1[2]
            print 'W1_3'
            print W1[3].shape
            print W1[3]
            print 'b1_0'
            print b1[0].shape
            print b1[0]
            print 'b1_1'
            print b1[1].shape
            print b1[1]
            print 'b1_2'
            print b1[2].shape
            print b1[2]
            print 'b1_3'
            print b1[3].shape
            print b1[3]
            print 'W2_0'
            print W2[0].shape
            print W2[0]
            print 'W2_1'
            print W2[1].shape
            print W2[1]
            print 'W2_2'
            print W2[2].shape
            print W2[2]
            print 'b2_0'
            print b2[0].shape
            print b2[0]
            print 'b2_1'
            print b2[1].shape
            print b2[1]
            print 'b2_2'
            print b2[2].shape
            print b2[2]
            print 'Wh'
            print Wh.shape
            print Wh
            print 'bh'
            print bh.shape
            print bh
            print 'Wo'
            print Wo.shape
            print Wh
            '''
             
            if iter%10 == 0: # TODO: change 2 to 50
                print('%d iter, loss = %.5f' %(iter, loss))
                sys.stdout.flush()
            if iter%FLAGS.eval_freq == 0:
                dev_map, dev_mrr = dev_evaluator.evaluate()
                test_map, test_mrr = test_evaluator.evaluate()
                if dev_map > best_dev_map:
                    not_improving = 0
                    best_dev_map, best_dev_mrr = dev_map, dev_mrr
                    best_test_map, best_test_mrr = test_map, test_mrr
                    best_iter = iter
                    print('New best valid MAP!')
                    saver.save(sess, FLAGS.save_path + '/model.tf')
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
        print('\n\nTraining finished!')
        print('***************')
        print('Best at iter %d' %best_iter)
        print('Performance dev: MAP %.3f   MRR %.3f <==' %(best_dev_map, best_dev_mrr))
        print('Performance test: MAP %.3f   MRR %.3f' %(best_test_map, best_test_mrr))

def main(argv):
    if FLAGS.dataset != 'TrecQA' and FLAGS.dataset != 'WikiQA':
        print('Error dataset!')
        sys.exit()
    if FLAGS.save_path!='':
        FLAGS.save_path = FLAGS.save_path
    else:
        save_path = [FLAGS.dataset]
        if FLAGS.dataset == 'TrecQA': 
            save_path.append(FLAGS.version)
        save_path.extend([FLAGS.model, FLAGS.update_tech, str(FLAGS.dim), str(FLAGS.filterA), 
                          str(FLAGS.filterB), str(FLAGS.lr), str(FLAGS.reg),
                          str(FLAGS.batch_size), str(FLAGS.max_length), str(FLAGS.keep_prob)])
        FLAGS.save_path = 'save_models/' + '_'.join(save_path)
        if not os.path.isdir(FLAGS.save_path):
            os.makedirs(FLAGS.save_path)
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    log_file = open(FLAGS.save_path + '/log', 'w')
    sys.stdout = log_file
    sys.stderr = log_file
    
    print '----CONFIGURATION----'
    print 'hostname=%s' %socket.gethostname()
    try:
        print 'CUDA_VISIBLE_DEVICES=%s' %os.environ["CUDA_VISIBLE_DEVICES"]
    except:
        print 'Warning: CUDA_VISIBLE_DEVICES was not specified'
    print 'dataset=%s' %FLAGS.dataset
    print 'version=%s' %FLAGS.version
    print 'model=%s' %FLAGS.model
    print 'dim=%d' %FLAGS.dim
    print 'filterA=%d' %FLAGS.filterA
    print 'filterB=%d' %FLAGS.filterB
    print 'lr=%f' %FLAGS.lr
    print 'reg=%f' %FLAGS.reg
    print 'keep_prob=%f' %FLAGS.keep_prob
    print 'update_tech=%s' %FLAGS.update_tech
    print 'batch_size=%d' %FLAGS.batch_size
    print 'max_length=%d' %FLAGS.max_length
    print 'eval_freq=%d' %FLAGS.eval_freq
    print 'sampling=%s' %FLAGS.sampling
    print 'load_model=%s' %FLAGS.load_model
    print 'save_path=%s' %FLAGS.save_path
    print '**************\n\n'
    sys.stdout.flush()

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

    train_dataset = read_relatedness_dataset(train_dir, vocab, debug=False) #TODO: change debug to false # This is a dict
    dev_dataset = read_relatedness_dataset(dev_dir, vocab, debug=False)
    test_dataset = read_relatedness_dataset(test_dir, vocab, debug=False)
    print('train_dir: %s, num train = %d' % (train_dir, train_dataset['size']))
    print('dev_dir: %s, num dev = %d' % (dev_dir, dev_dataset['size']))
    print('test_dir: %s, num test = %d' % (test_dir, test_dataset['size']))

    train(train_dataset, dev_dataset, test_dataset, vecs)
    sys.stdout = orig_stdout
    sys.stderr = orig_stderr
    log_file.close()

if __name__ == '__main__':
    tf.app.flags.DEFINE_string('dataset', 'TrecQA', 'dataset, can be TrecQA or WikiQA')
    tf.app.flags.DEFINE_string('version', 'raw', 'the version of TrecQA dataset, can be raw and clean')
    #tf.app.flags.DEFINE_string('model', 'average-pointwise', 'the version of model to be used')
    tf.app.flags.DEFINE_string('model', 'mpssn-pointwise', 'the version of model to be used')

    tf.app.flags.DEFINE_integer('dim', 150, 'dimension of hidden layers')
    tf.app.flags.DEFINE_integer('filterA', 50, 'number of filter A')
    tf.app.flags.DEFINE_integer('filterB', 10, 'number of filter B')
    tf.app.flags.DEFINE_float('lr', 0.001, 'learning rate')
    tf.app.flags.DEFINE_float('reg', 0.01, 'regularization weight')
    tf.app.flags.DEFINE_float('keep_prob', 0.5, 'keep probability of dropout during training')
    tf.app.flags.DEFINE_string('update_tech', 'adam', 'gradient descent technique')



    tf.app.flags.DEFINE_integer('batch_size', 64, 'mini-batch size') #TODO: change batch size to 64
    tf.app.flags.DEFINE_integer('max_length', 48, 'max sentence length')
    tf.app.flags.DEFINE_integer('eval_freq', 1000, 'evaluate every x batches')
    tf.app.flags.DEFINE_string('sampling', 'max', 'sampling strategy, max or random')
    tf.app.flags.DEFINE_string('load_model', '', 'specify a path to load a model')
    tf.app.flags.DEFINE_string('save_path', '', 'specify a path to save the best model')
    
    tf.app.run()
