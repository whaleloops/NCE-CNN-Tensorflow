import tensorflow as tf
import numpy as np

class SentencePairEncoder(object):
    def __init__(self, pretrained_embeddings, dim=100,
                 use_tanh=False, verbose=False, 
                 dropout_keep=1.0, seq_length=32):
        self._vocab_size = pretrained_embeddings.shape[0]
        self._embed_dim = pretrained_embeddings.shape[1]
        self._dim = dim
        self._non_linear = tf.nn.tanh if use_tanh else tf.nn.relu
        self._verbose = verbose
        self._dropout_keep = dropout_keep
        self._seq_length = seq_length
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        
        self.input_questions = tf.placeholder(tf.int32, [None, seq_length], name='input_questions')
        self.input_question_lens = tf.placeholder(tf.int32, [None], name='input_question_lens')
        self.input_answers = tf.placeholder(tf.int32, [None, seq_length], name='input_answers')
        self.input_answer_lens = tf.placeholder(tf.int32, [None], name='input_answer_lens')
        self.keep_prob = tf.placeholder(tf.float32)
        self.word_embeddings = tf.get_variable(name='word_embeddings',
                                               initializer=tf.constant(pretrained_embeddings, dtype=tf.float32),
                                               trainable=True)
        self.init_extra()
        self.derive_loss()

    def init_extra(self):
        '''
        This function could be override in a child class 
        for extra placeholders or parameters.
        '''
        self.labels = tf.placeholder(tf.int32, [None], name='labels')
        self.linear_matrix = tf.get_variable(name='linear_matrix',
                                               shape=[self._embed_dim * 2, self._dim],
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               trainable=True)
        self.score_vector = tf.get_variable(name='score_vector',
                                               shape=[self._dim, 1],
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               trainable=True)
    
    def produce_feature(self, sentences, lens):
        '''
        This function could be overrided in a child class
        to generate qa pair features from word embedding layer.
        '''
        #(batchsize, seq_length, embed_size)
        embed_layer = tf.nn.embedding_lookup(self.word_embeddings, sentences)
        embed_layer_mask = tf.expand_dims(tf.cast(tf.sequence_mask(lens, self._seq_length), tf.float32), 2) * embed_layer
        features = tf.reduce_sum(embed_layer_mask, axis=1) / tf.expand_dims(tf.cast(lens, tf.float32), 1)
        
        #(batchsize, 1, embed_size, 1)
        '''
        q_features = tf.nn.pool(q_embed_layer,
                                      window_shape = [1, self._seq_length, 1, 1], 
                                      strides = [1, self._seq_length, 1, 1],
                                      pooling_type = 'AVG',
                                      padding = 'SAME',
                                      name = 'word_embedding_pooling')
        a_features = tf.nn.pool(a_embed_layer,
                                      window_shape = [1, self._seq_length, 1, 1], 
                                      strides = [1, self._seq_length, 1, 1],
                                      pooling_type = 'AVG',
                                      padding = 'SAME',
                                      name = 'word_embedding_pooling')
        '''
        return features
    
    def derive_loss(self):
        '''
        This function could be overided in a child class
        to derive score and loss.
        '''
        q_features = self.produce_feature(self.input_questions, self.input_question_lens) 
        a_features = self.produce_feature(self.input_answers, self.input_answer_lens) 
        pair_features = tf.concat([q_features, a_features], axis=1)
        pair_features = tf.nn.relu(tf.matmul(pair_features, self.linear_matrix))#(batchsize, last_dim)
        pair_features = tf.nn.dropout(pair_features, self.keep_prob)
        scores = tf.matmul(pair_features, self.score_vector)#(batchsize, 1)
        self.scores = tf.reshape(scores, [-1])#(batchsize,)
        probs = tf.sigmoid(self.scores)
        losses = - tf.cast(self.labels, tf.float32) * tf.log(probs + 1e-8) \
                 - (1.0 - tf.cast(self.labels, tf.float32)) * tf.log(1.0 - probs + 1e-8)
        self.loss = tf.reduce_mean(losses)
        #l2_loss = tf.constant(l2_weight) * tf.nn.l2_loss()
        

class SentencePairEncoderCNN(SentencePairEncoder):
    def init_extra(self):
        '''
        This function could be override in a child class 
        for extra placeholders or parameters.
        '''
        self.labels = tf.placeholder(tf.int32, [None], name='labels')
        self.linear_matrix = tf.get_variable(name='linear_matrix',
                                               shape=[self._dim * 2, self._dim],
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               trainable=True)
        self.score_vector = tf.get_variable(name='score_vector',
                                               shape=[self._dim, 1],
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               trainable=True)
        
        self.W_conv1 = tf.get_variable(name='W_conv1',
                                  shape=[2, self.word_embeddings.shape[1], 1, self._dim],
                                  initializer=tf.contrib.layers.xavier_initializer(),
                                  trainable=True)
        self.b_conv1 = tf.get_variable(name='b_conv1', 
                                  initializer=tf.constant(np.array([0.1] * self._dim, dtype=np.float32)),
                                  trainable=True)
    
    def produce_feature(self, sentences, lens):
        '''
        This function could be overrided in a child class
        to generate qa pair features from word embedding layer.
        '''
        #(batchsize, seq_length, embed_size)
        embed_layer = tf.nn.embedding_lookup(self.word_embeddings, sentences)
        #(batchsize, seq_length, embed_size) 
        embed_layer_mask = tf.expand_dims(tf.cast(tf.sequence_mask(lens, self._seq_length), tf.float32), 2) * embed_layer
        #(batchsize, seq_length, embed_size, 1)
        embed_layer_mask = tf.expand_dims(embed_layer_mask, 3)
        #(batchsize, -1, 1, dim)
        layer_conv1 = tf.nn.conv2d(embed_layer_mask, self.W_conv1, strides=[1,1,1,1], padding="VALID",name="conv1")
        #(batchsize, -1, 1, dim)
        layer_conv1 = self._non_linear(layer_conv1 + self.b_conv1)
        #(batchsize, 1, dim)
        layer_pool1 = tf.reduce_max(layer_conv1, axis=1)
        #(batchsize, dim)
        features = tf.squeeze(layer_pool1, 1)
        
        return features

class SentencePairEncoderPairwiseRanking(SentencePairEncoder):
    def init_extra(self):
        '''
        This function could be override in a child class 
        for extra placeholders or parameters.
        '''
        self.neg_answers = tf.placeholder(tf.int32, [None, self._seq_length], name='neg_answers')
        self.neg_answer_lens = tf.placeholder(tf.int32, [None], name='neg_answer_lens')
        self.linear_matrix = tf.get_variable(name='linear_matrix',
                                               shape=[self._embed_dim * 2, self._dim],
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               trainable=True)
        self.score_vector = tf.get_variable(name='score_vector',
                                               shape=[self._dim, 1],
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               trainable=True)
    
    def derive_loss(self):
        '''
        This function could be overided in a child class
        to derive score and loss.
        '''
        q_features = self.produce_feature(self.input_questions, self.input_question_lens) 
        pos_a_features = self.produce_feature(self.input_answers, self.input_answer_lens) 
        neg_a_features = self.produce_feature(self.neg_answers, self.neg_answer_lens)
         
        pos_pair_features = tf.concat([q_features, pos_a_features], axis=1)
        pos_pair_features = tf.nn.relu(tf.matmul(pos_pair_features, self.linear_matrix))#(batchsize, last_dim)
        pos_pair_features = tf.nn.dropout(pos_pair_features, self.keep_prob)
        pos_scores = tf.matmul(pos_pair_features, self.score_vector)#(batchsize, 1)
        self.scores = tf.reshape(pos_scores, [-1])#(batchsize,)
        neg_pair_features = tf.concat([q_features, neg_a_features], axis=1)
        neg_pair_features = tf.nn.relu(tf.matmul(neg_pair_features, self.linear_matrix))#(batchsize, last_dim)
        neg_pair_features = tf.nn.dropout(neg_pair_features, self.keep_prob)
        neg_scores = tf.matmul(neg_pair_features, self.score_vector)#(batchsize, 1)
        neg_scores = tf.reshape(neg_scores, [-1])#(batchsize,)
        losses = tf.maximum(0.0, 1 - self.scores + neg_scores)
        self.loss = tf.reduce_mean(losses)
    
            
class SentencePairEncoderPairwiseRankingCNN(SentencePairEncoderPairwiseRanking):
    def init_extra(self):
        '''
        This function could be override in a child class 
        for extra placeholders or parameters.
        '''
        self.neg_answers = tf.placeholder(tf.int32, [None, self._seq_length], name='neg_answers')
        self.neg_answer_lens = tf.placeholder(tf.int32, [None], name='neg_answer_lens')
        self.linear_matrix = tf.get_variable(name='linear_matrix',
                                               shape=[18 * self._dim, self._dim],
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               trainable=True)
        self.score_vector = tf.get_variable(name='score_vector',
                                               shape=[self._dim, 1],
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               trainable=True)
        self.W_conv1 = tf.get_variable(name='W_conv1',
                                  shape=[1, self.word_embeddings.shape[1], 1, self._dim],
                                  initializer=tf.contrib.layers.xavier_initializer(),
                                  trainable=True)
        self.b_conv1 = tf.get_variable(name='b_conv1', 
                                  initializer=tf.constant(np.array([0.1] * self._dim, dtype=np.float32)),
                                  trainable=True)
        self.W_conv2 = tf.get_variable(name='W_conv2',
                                  shape=[2, self.word_embeddings.shape[1], 1, self._dim],
                                  initializer=tf.contrib.layers.xavier_initializer(),
                                  trainable=True)
        self.b_conv2 = tf.get_variable(name='b_conv2', 
                                  initializer=tf.constant(np.array([0.1] * self._dim, dtype=np.float32)),
                                  trainable=True)
        self.W_conv3 = tf.get_variable(name='W_conv3',
                                  shape=[3, self.word_embeddings.shape[1], 1, self._dim],
                                  initializer=tf.contrib.layers.xavier_initializer(),
                                  trainable=True)
        self.b_conv3 = tf.get_variable(name='b_conv3', 
                                  initializer=tf.constant(np.array([0.1] * self._dim, dtype=np.float32)),
                                  trainable=True)
    
    def produce_feature(self, sentences, lens):
        '''
        This function could be overrided in a child class
        to generate qa pair features from word embedding layer.
        '''
        #(batchsize, seq_length, embed_size)
        embed_layer = tf.nn.embedding_lookup(self.word_embeddings, sentences)
        #(batchsize, seq_length, embed_size) 
        embed_layer_mask = tf.expand_dims(tf.cast(tf.sequence_mask(lens, self._seq_length), tf.float32), 2) * embed_layer
        #(batchsize, seq_length, embed_size, 1)
        embed_layer_mask = tf.expand_dims(embed_layer_mask, 3)
        #(batchsize, seq_length, 1, dim)
        layer_conv1 = tf.nn.conv2d(embed_layer_mask, self.W_conv1, strides=[1,1,self.word_embeddings.shape[1],1], padding="SAME",name="conv1")
        #(batchsize, seq_length, 1, dim)
        layer_conv1 = self._non_linear(layer_conv1 + self.b_conv1)
        #(batchsize, seq_length, 1, dim)
        layer_conv2 = tf.nn.conv2d(embed_layer_mask, self.W_conv2, strides=[1,1,self.word_embeddings.shape[1],1], padding="SAME",name="conv2")
        #(batchsize, seq_length, 1, dim)
        layer_conv2 = self._non_linear(layer_conv2 + self.b_conv2)
        #(batchsize, seq_length, 1, dim)
        layer_conv3 = tf.nn.conv2d(embed_layer_mask, self.W_conv3, strides=[1,1,self.word_embeddings.shape[1],1], padding="SAME",name="conv3")
        #(batchsize, seq_length, 1, dim)
        layer_conv3 = self._non_linear(layer_conv3 + self.b_conv3)
        #(batchsize, seq_length, dim)
        layer_conv1 = tf.squeeze(layer_conv1, 2)
        layer_conv2 = tf.squeeze(layer_conv2, 2)
        layer_conv3 = tf.squeeze(layer_conv3, 2)
        #(batchsize, seq_length, dim)
        layer_conv1 = tf.expand_dims(tf.cast(tf.sequence_mask(lens, self._seq_length), tf.float32), 2) * layer_conv1
        layer_conv2 = tf.expand_dims(tf.cast(tf.sequence_mask(lens, self._seq_length), tf.float32), 2) * layer_conv2
        layer_conv3 = tf.expand_dims(tf.cast(tf.sequence_mask(lens, self._seq_length), tf.float32), 2) * layer_conv3
        #(batchsize, dim)
        layer_pool1_max = tf.reduce_max(layer_conv1, axis=1)
        layer_pool1_mean = tf.reduce_mean(layer_conv1, axis=1)
        layer_pool1_min = tf.reduce_min(layer_conv1, axis=1)
        layer_pool2_max = tf.reduce_max(layer_conv2, axis=1)
        layer_pool2_mean = tf.reduce_mean(layer_conv2, axis=1)
        layer_pool2_min = tf.reduce_min(layer_conv2, axis=1)
        layer_pool3_max = tf.reduce_max(layer_conv3, axis=1)
        layer_pool3_mean = tf.reduce_mean(layer_conv3, axis=1)
        layer_pool3_min = tf.reduce_min(layer_conv3, axis=1)
        #9 * (batchsize, dim)
        features = tf.concat([layer_pool1_max, layer_pool1_mean, layer_pool1_min,
                              layer_pool2_max, layer_pool2_mean, layer_pool2_min,
                              layer_pool3_max, layer_pool3_mean, layer_pool3_min], axis=1)
        #features = [layer_pool1_max, layer_pool1_mean, layer_pool1_min,
        #            layer_pool2_max, layer_pool2_mean, layer_pool2_min,
        #            layer_pool3_max, layer_pool3_mean, layer_pool3_min]
        return features
    
    def derive_loss(self):
        '''
        This function could be overided in a child class
        to derive score and loss.
        '''
        q_features = self.produce_feature(self.input_questions, self.input_question_lens) 
        pos_a_features = self.produce_feature(self.input_answers, self.input_answer_lens) 
        neg_a_features = self.produce_feature(self.neg_answers, self.neg_answer_lens)
        '''
        pos_feature_list = []
        neg_feature_list = []
        
        for i in range(len(q_features)):
            #(batchsize, 1)
            pos_cos_dis = tf.reduce_sum(q_features[i] * pos_a_features[i], 1, keep_dims=True)
            #(batchsize, 1)
            pos_l2_dis = tf.norm(q_features[i] - pos_a_features[i], ord=2, axis=1, keep_dims=True)
            pos_l1_dis = tf.norm(q_features[i] - pos_a_features[i], ord=1, axis=1, keep_dims=True)
            pos_feature_list.extend([pos_cos_dis, pos_l2_dis, pos_l1_dis])
            neg_cos_dis = tf.reduce_sum(q_features[i] * neg_a_features[i], 1, keep_dims=True)
            #(batchsize, 1)
            neg_l2_dis = tf.norm(q_features[i] - neg_a_features[i], ord=2, axis=1, keep_dims=True)
            neg_l1_dis = tf.norm(q_features[i] - neg_a_features[i], ord=1, axis=1, keep_dims=True)
            neg_feature_list.extend([neg_cos_dis, neg_l2_dis, neg_l1_dis])
        '''
        #(batchsize, 27)
        #pos_pair_features = tf.concat(pos_feature_list, axis=1)
        pos_pair_features = tf.concat([q_features, pos_a_features], axis=1)
        pos_pair_features = tf.nn.dropout(pos_pair_features, self.keep_prob)
        pos_pair_features = tf.nn.relu(tf.matmul(pos_pair_features, self.linear_matrix))#(batchsize, last_dim)
        pos_pair_features = tf.nn.dropout(pos_pair_features, self.keep_prob)
        pos_scores = tf.matmul(pos_pair_features, self.score_vector)#(batchsize, 1)
        self.scores = tf.reshape(pos_scores, [-1])#(batchsize,)
        #(batchsize, 27)
        #neg_pair_features = tf.concat(neg_feature_list, axis=1)
        neg_pair_features = tf.concat([q_features, neg_a_features], axis=1)
        neg_pair_features = tf.nn.dropout(neg_pair_features, self.keep_prob)
        neg_pair_features = tf.nn.relu(tf.matmul(neg_pair_features, self.linear_matrix))#(batchsize, last_dim)
        neg_pair_features = tf.nn.dropout(neg_pair_features, self.keep_prob)
        neg_scores = tf.matmul(neg_pair_features, self.score_vector)#(batchsize, 1)
        neg_scores = tf.reshape(neg_scores, [-1])#(batchsize,)
        losses = tf.maximum(0.0, 1 - self.scores + neg_scores)
        l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in tf.trainable_variables() if 'word_embeddings' not in w.name])
        self.loss = tf.reduce_mean(losses) + 0.01 * l2_loss

class SentencePairEncoderPairwiseRankingGRU(SentencePairEncoderPairwiseRanking):
    def init_extra(self):
        '''
        This function could be override in a child class 
        for extra placeholders or parameters.
        '''
        self.neg_answers = tf.placeholder(tf.int32, [None, self._seq_length], name='neg_answers')
        self.neg_answer_lens = tf.placeholder(tf.int32, [None], name='neg_answer_lens')
        #'''
        self.linear_matrix = tf.get_variable(name='linear_matrix',
                                               shape=[2 * self._dim, self._dim],
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               trainable=True)
        self.score_vector = tf.get_variable(name='score_vector',
                                               shape=[self._dim, 1],
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               trainable=True)
        #'''
        #self.forback_combine_matrix = tf.get_variable(name='combine_matrix',
        #                                       shape=[2 * self._dim, self._dim],
        #                                       initializer=tf.contrib.layers.xavier_initializer(),
        #                                       trainable=True)
        #self.forward_rnn = tf.contrib.rnn.GRUCell(self._dim)
        forward_rnn_layers = [tf.contrib.rnn.GRUCell(self._dim) for _ in range(1)]
        self.forward_rnn = tf.contrib.rnn.MultiRNNCell(forward_rnn_layers)
        #self.backward_rnn = tf.contrib.rnn.GRUCell(self._dim)
        #backward_rnn_layers = [tf.contrib.rnn.GRUCell(self._dim) for _ in range(1)]
        #self.backward_rnn = tf.contrib.rnn.MultiRNNCell(backward_rnn_layers)
    
    def produce_feature(self, sentences, lens):
        '''
        This function could be overrided in a child class
        to generate qa pair features from word embedding layer.
        '''
        #(batchsize, seq_length, embed_size)
        embed_layer = tf.nn.embedding_lookup(self.word_embeddings, sentences)
        #(batchsize, seq_length, embed_size) 
        embed_layer_mask = tf.expand_dims(tf.cast(tf.sequence_mask(lens, self._seq_length), tf.float32), 2) * embed_layer
        #(batchsize, seq_length, dim)
        #outputs, _ = tf.nn.bidirectional_dynamic_rnn(self.forward_rnn, self.backward_rnn, embed_layer_mask, sequence_length = lens, dtype=tf.float32)
        outputs, _ = tf.nn.dynamic_rnn(self.forward_rnn, embed_layer_mask, sequence_length = lens, dtype=tf.float32)
        #(batchsize, seq_length, dim)
        #forward_outputs, backward_outputs = outputs
        forward_outputs = outputs
        #(batchsize, seq_length, dim)
        forward_outputs_mask = tf.expand_dims(tf.cast(tf.sequence_mask(lens, self._seq_length), tf.float32), 2) * forward_outputs
        #backward_outputs_mask = tf.expand_dims(tf.cast(tf.sequence_mask(lens, self._seq_length), tf.float32), 2) * backward_outputs
        #(batchsize, seq_length, 2*dim)
        #hidden_states = tf.concat([forward_outputs_mask, backward_outputs_mask], 2)
        #hidden_states = tf.reshape(hidden_states, [-1, 2 * self._dim])
        #(batchsize, seq_length, dim)
        #hidden_states = self._non_linear(tf.matmul(hidden_states, self.forback_combine_matrix))
        #hidden_states = tf.reshape(hidden_states, [-1, self._seq_length, self._dim])
        #(batchsize, dim)
        #features = tf.reduce_sum(hidden_states, axis=1) / tf.expand_dims(tf.cast(lens, tf.float32), 1)
        features = tf.reduce_sum(forward_outputs_mask, axis=1) / tf.expand_dims(tf.cast(lens, tf.float32), 1)
        #features = tf.reduce_max(hidden_states, axis=1)
        return features
    
    def derive_loss(self):
        '''
        This function could be overided in a child class
        to derive score and loss.
        '''
        with tf.variable_scope('RNN', reuse=None):
            q_features = self.produce_feature(self.input_questions, self.input_question_lens) 
        with tf.variable_scope('RNN', reuse=True):
            pos_a_features = self.produce_feature(self.input_answers, self.input_answer_lens) 
            neg_a_features = self.produce_feature(self.neg_answers, self.neg_answer_lens)
        pos_pair_features = tf.concat([q_features, pos_a_features], axis=1)
        pos_pair_features = tf.nn.relu(tf.matmul(pos_pair_features, self.linear_matrix))#(batchsize, last_dim)
        pos_pair_features = tf.nn.dropout(pos_pair_features, self.keep_prob)
        pos_scores = tf.matmul(pos_pair_features, self.score_vector)#(batchsize, 1)
        #pos_scores = tf.reduce_sum(q_features * pos_a_features, 1, keep_dims=True)
        self.scores = tf.reshape(pos_scores, [-1])#(batchsize,)
        neg_pair_features = tf.concat([q_features, neg_a_features], axis=1)
        neg_pair_features = tf.nn.relu(tf.matmul(neg_pair_features, self.linear_matrix))#(batchsize, last_dim)
        neg_pair_features = tf.nn.dropout(neg_pair_features, self.keep_prob)
        neg_scores = tf.matmul(neg_pair_features, self.score_vector)#(batchsize, 1)
        #neg_scores = tf.reduce_sum(q_features * neg_a_features, 1, keep_dims=True)
        neg_scores = tf.reshape(neg_scores, [-1])#(batchsize,)
        losses = tf.maximum(0.0, 1 - self.scores + neg_scores)
        l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in tf.trainable_variables() if 'word_embeddings' not in w.name])
        self.loss = tf.reduce_mean(losses) + 0.01 * l2_loss
