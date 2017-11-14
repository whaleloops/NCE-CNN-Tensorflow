#%%
# 2.1 Import libraries.
import gc
import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

import matplotlib.pyplot as plt


#init
x = np.array([[1,2],[3,4],[5,6],[7,8]])
y = np.array([7,5,3,1])
print x

#tf
x_placeholder = tf.placeholder(tf.float32, name='x')
y_placeholder = tf.placeholder(tf.float32, name='y')
slice_op = x_placeholder[:,1]
add_op = tf.add(y_placeholder, slice_op, name='add')

#%%
# 2.7 Run training for MAX_STEPS and save checkpoint at the end.
with tf.Session() as sess:
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    tf_result = sess.run(add_op, feed_dict={x_placeholder: x, y_placeholder: y})

print tf_result
print tf_result.shape