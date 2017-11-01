#%%
# 2.1 Import libraries.
import gc
import os
import time, datetime
from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
import tensorflow as tf

#%%
# 2.2 Define some constants.

#%%
# 2.3 Get input data.

#%%
# 2.6 Build the complete graph for feeding inputs, training, and saving checkpoints.
cnn_graph = tf.Graph()
m1 = createModel() # TODO: add more
with cnn_graph.as_default():
    # Generate placeholders for the images and labels. #TODO: change this
    images_placeholder = tf.placeholder(tf.float32)
    labels_placeholder = tf.placeholder(tf.int32)

    # Build a Graph that computes predictions from the inference model.
    logits = m1.cnn_inference()

    # Add to the Graph the Ops that calculate and apply gradients.
    learning_rate = 0.01 #TODO: change
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

  # Start the training loop.
  for step in xrange(MAX_STEPS):
    # Read a batch of images and labels. #TODO: change this
    x1 = Xtrain[0][i:i + conf.batch_size]
    x2 = Xtrain[1][i:i + conf.batch_size]
    y = ytrain[i:i + conf.batch_size]

    # Run one step of the model.  The return values are the activations
    # from the `train_op` (which is discarded) and the `loss` Op.  To
    # inspect the values of your Ops or variables, you may include them
    # in the list passed to sess.run() and the value tensors will be
    # returned in the tuple from the call. #TODO: change this
    _, summaries, accc, loss = sess.run[train_op, train_summary_op, acc, cost],
                          feed_dict={input_1: x1, input_2: x2, input_3: y, dropout_keep_prob: 1.0})

    time_str = datetime.datetime.now().isoformat()
    print("{}: loss {:g}, acc {:g}".format(time_str, loss, accc))
    train_summary_writer.add_summary(summaries)

  # Write a checkpoint.
  train_writer.close()
  checkpoint_file = os.path.join(MODEL_SAVE_PATH, 'checkpoint')
  saver.save(sess, checkpoint_file, global_step=step)

#%%
# 2.8 Run evaluation based on the saved checkpoint. (One example)