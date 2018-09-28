import tensorflow as tf
import numpy as np

class RvNN():
    def __int__(self):
        self.dimension = 0
        self.rule_length = 0
        self.training_iteration = 0

    def init(self, dim, rule_length, training_iteration):
        self.dimension = dim
        self.rule_length = rule_length
        self.training_iteration = training_iteration

    def add_layer(self, input, activation_function=None):
        regularizion = tf.contrib.layers.l2_regularizer(0.0001)  # L2 regularizion
        with tf.variable_scope('Composition', initializer=tf.random_normal_initializer,
                               regularizer=regularizion):
            weight = tf.get_variable('weight', shape=[self.dimension, self.dimension*2])
            bias = tf.get_variable('bias', shape=[self.dimension, 1])
        Wx_plus_b = tf.matmul(weight, input) + bias
        if activation_function is None:
            output = Wx_plus_b
        else:
            output = activation_function(Wx_plus_b)
        return output

    def run(self):
        # define the placeholder for inputs to network
        x_c1 = tf.placeholder(tf.float32, [None, self.dimension])
        x_c2 = tf.placeholder(tf.float32, [None, self.dimension])
        x = tf.concat([x_c1, x_c2], 0)

        # add hidden layer with the length of rule
        for i in range(self.rule_length-1):
            layer = self.add_layer(x, tf.tanh)
        loss = 0  # how to calculate?
        train_step = tf.train.AdagradOptimizer(0.1).minimize(loss)  # learning rate could be halved?
        init_op = tf.global_variables_initializer()

        sess = tf.Session()
        sess.run(init_op)
        for i in range(self.training_iteration):
            # batch to train
            pass

def test_RvNN():
    model = RvNN()
    # init(self, dim, rule_length, training_iteration)
    # batch_size
    model.init(100, 2, 1000)

if __name__ == "__main__":
    test_RvNN()
