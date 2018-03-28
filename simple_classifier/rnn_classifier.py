import numpy as np
import tensorflow as tf

class RnnClassifier:
    def __init__(self, vocab, n_topics):
        
        self.words = vocab.words
        
        self.embeddings = vocab.embeddings
        self.vocab_size, self.embedding_dim = self.embeddings.shape
        self.max_sent_length = vocab.max_sent_length
        
        self.graph = tf.Graph()
        
        self.n_topics = n_topics
        with self.graph.as_default():
            self.create_graph()
            self.saver = tf.train.Saver()
            
            
    def create_graph(self):
    
        
        self.inp = tf.placeholder(tf.int32, [None, None])
        self.mask = tf.placeholder(tf.float64, [None, None])
        with tf.variable_scope("input"):
            self.word_embeddings = tf.get_variable(initializer=tf.constant(self.embeddings),
                    trainable=False, name="word_embeddings")
            embeds = tf.nn.embedding_lookup(self.word_embeddings, self.inp)
        
        print(embeds)
        self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=128)
#         print(embeds)
        
        outputs, unrolled = tf.nn.dynamic_rnn(self.rnn_cell, embeds, dtype=tf.float64)
        
#         summed = tf.reduce_sum(embeds, axis=1, name='sum')/tf.reshape(tf.reduce_sum(self.mask, axis=1), shape=(-1, 1))
        
        print(unrolled)
        
        dense = tf.layers.dense(unrolled.h, units=256, name='dense_1')
        
        self.logits = tf.layers.dense(dense, units=self.n_topics, name='logits')
        
        input_shape = tf.shape(self.inp)
        self.inp_y = tf.placeholder(tf.float32, [None, self.n_topics])
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.inp_y, logits=self.logits, name='cross_entropy'))
        
        self.probs = tf.nn.softmax(self.logits)
        
        self.optimizer = tf.train.AdamOptimizer()
        
        self.optimization_step = self.optimizer.minimize(self.loss)
        
        