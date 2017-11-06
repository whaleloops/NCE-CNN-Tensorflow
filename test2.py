#%%
# 2.1 Import libraries.
import gc
import os
import time, datetime
from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
import tensorflow as tf
import embedding as emb

import argparse
import sys

from util.Vocab import Vocab
from util.read_data import read_relatedness_dataset, read_embedding
from data_generate import DataGeneratePointWise

#%%
# 2.2 Define some constants.
NUM_CLASSES = 2;
MAX_SENT_LENGTH_CONV = 25;
REG_LAMBDA = 1e-4;
BATCH_SIZE = 32;
N_HIDDEN = 150;
num_filters_A = 300;
num_filters_B = 20;
filter_size = [1,2,100]
#%%
# 2.3 Get input data.
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

#%% main
if __name__ == '__main__':
  # Configure the argument parser
  parser = argparse.ArgumentParser(description = 'Python to run pairwise training for QA')
  parser.add_argument('--dataset', action='store', dest='dataset', default='TrecQA',
                      help='dataset, can be TrecQA or WikiQA')
  parser.add_argument('--version', action='store', dest='version', default='raw',
                      help='the version of TrecQA dataset, can be raw and clean')
  parser.add_argument('--num_pairs', action='store', dest='num_pairs', type=int, default=8,
                      help='number of negative samples for each pos sample')
  parser.add_argument('--neg_mode', action='store', dest='neg_mode', type=int, default=2,
                      help='negative sample strategy, 1 is random sampling, 2 ismax sampling and 3 is mix sampling')
  opt = parser.parse_args()
  # pid = arguments.a

  # read default arguments 
  args = {
    'model' : 'pairwise-conv', #convolutional neural network 
    'layers' : 1, # number of hidden layers in the fully-connected layer
    'dim' : 150, # number of neurons in the hidden layer.
    'dropout_mode' : 1 # add dropout by default, to turn off change its value to 0
  }

  model_name = 'pairwise-conv'
  # model_class = similarityMeasure.Conv # TODO class name
  model_structure = model_name

  # torch.seed() TODO0
  # torch.manualSeed(-3.0753778015266e+18)
  # print('<torch> using the automatic seed: ' .. torch.initialSeed())

  if opt.dataset != 'TrecQA' and opt.dataset != 'WikiQA' and opt.dataset != 'msrvid':
    print('Error dataset!')
    sys.exit()

  # directory containing dataset files
  data_dir = 'data/' + opt.dataset + '/'

  # load vocab
  vocab = Vocab(data_dir + 'vocab.txt')

  # load embeddings
  print('loading glove word embeddings')
  vecs = load_glove_word_embeddings(vocab)
  taskD = 'vid'

  # load datasets
  print('loading datasets' + opt.dataset)
  if opt.dataset == 'TrecQA':
    train_dir = data_dir + 'train-all/'
    dev_dir = data_dir + opt.version + '-dev/'
    test_dir = data_dir + opt.version + '-test/'
  elif opt.dataset == 'WikiQA':
    train_dir = data_dir + 'train/'
    dev_dir = data_dir + 'dev/'
    test_dir = data_dir + 'test/'
  elif opt.dataset == 'msrvid':
    train_dir = data_dir + 'train/'
    dev_dir = data_dir + 'dev/'
    test_dir = data_dir + 'test/'    

  train_dataset = read_relatedness_dataset(train_dir, vocab, taskD) # This is a dict
  dev_dataset = read_relatedness_dataset(dev_dir, vocab, taskD)
  print('train_dir: %s, num train = %d\n' % (train_dir, train_dataset['size']))
  print('dev_dir: %s, num dev   = %d\n' % (dev_dir, dev_dataset['size']))
  

  #%%
  # 2.6 Build the complete graph for feeding inputs, training, and saving checkpoints.
  cnn_graph = tf.Graph()

  with cnn_graph.as_default():
      # Generate placeholders for the images and labels.
      input_1 = tf.placeholder(tf.int32, [None, MAX_SENT_LENGTH_CONV], name="input_x1")
      input_2 = tf.placeholder(tf.int32, [None, MAX_SENT_LENGTH_CONV], name="input_x2")
      input_3 = tf.placeholder(tf.int32, [None, NUM_CLASSES], name="input_y")
      dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

      # Build a Graph that computes predictions from the inference model. TODO: Change
      m1 = MPSSN(NUM_CLASSES, embedding_size, filter_size, [num_filters_A, num_filters_B], N_HIDDEN,
             input_x1, input_x2, input_y, REG_LAMBDA, dropout_keep_prob)
      logits = m1.cnn_inference()

      # Add to the Graph the Ops that calculate and apply gradients.
      learning_rate = 0.01 
      train_op, acc, cost = m1.cnn_train(logits, learning_rate)

      # Add the variable initializer Op.
      init = tf.global_variables_initializer()

      # Create a saver for writing training checkpoints.
      saver = tf.train.Saver()

  #%%
  # 2.7 Run training for MAX_STEPS and save checkpoint at the end.
  with tf.Session(graph=cnn_graph) as sess:
    # Merge tf.summary
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))
    loss_summary = tf.summary.scalar("loss", cost)
    acc_summary = tf.summary.scalar("accuracy", acc)
    train_summary_op = tf.summary.merge([loss_summary, acc_summary])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Run the Op to initialize the variables.
    sess.run(init)
    #TODO
    data_generator = DataGeneratePointWise(model, sess, train_dataset, BATCH_SIZE, MAX_SENT_LENGTH_CONV, 'random')

    # Start the training loop.
    for step in xrange(MAX_STEPS):
      # Read a batch of images and labels.
      tmp = data_generator.next();

      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, summaries, accc, loss = sess.run[train_op, train_summary_op, acc, cost],
                            feed_dict={input_1: tmp.input_questions, input_2: tmp.input_answers, input_3: tmp.labels, dropout_keep_prob: 1.0})

      time_str = datetime.datetime.now().isoformat()
      print("{}: loss {:g}, acc {:g}".format(time_str, loss, accc))
      train_summary_writer.add_summary(summaries)

    # Write a checkpoint.
    train_writer.close()
    checkpoint_file = os.path.join(MODEL_SAVE_PATH, 'checkpoint')
    saver.save(sess, checkpoint_file, global_step=step)

  #%%
  # 2.8 Run evaluation based on the saved checkpoint. (One example)