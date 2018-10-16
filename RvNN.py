import tensorflow as tf
import numpy as np


class TreeNode:
    def __init__(self, embbeding):
        self.index = -1
        self.emb = embbeding  # the embedding of the predicate
        self.parent = None
        self.left = None
        self.right = None
        self.isLeaf = False


class RvNN:
    def __int__(self):
        self.dimension = 0
        self.rule_length = 0
        self.training_iteration = 0

    def init(self, dim, rule_length, training_iteration):
        self.dimension = dim
        self.rule_length = rule_length
        self.training_iteration = training_iteration

    # not finished!!!!!!!!!!!!!!!!!!!!!!
    def load_data(self):
        self.train_data = []
        self.test_data = 0

    # 输入参数，输出一整颗组装好的树结构的root节点
    # not finished!!!!!!!!!!!!!!!!!!!!!!
    def parse(self, data):
        root = TreeNode(data)  # data includes embedding
        return root

    def add_layers(self, node, activation_function=tf.tanh):  # tf.sigmoid 待定
        if not node.isLeaf:
            self.add_layers(node.left)

            # define the placeholder for inputs to network
            x_c1 = tf.placeholder(tf.float32, [None, self.dimension])
            x_c2 = tf.placeholder(tf.float32, [None, self.dimension])
            x = tf.concat([x_c1, x_c2], 0)

            regularizion = tf.contrib.layers.l2_regularizer(0.0001)  # L2 regularizion
            with tf.variable_scope('Composition', initializer=tf.random_normal_initializer,
                                   regularizer=regularizion):
                weight = tf.get_variable('weight', shape=[self.dimension, self.dimension * 2])
                bias = tf.get_variable('bias', shape=[self.dimension, 1])
            Wx_plus_b = tf.matmul(weight, x) + bias
            if activation_function is None:
                output = Wx_plus_b
            else:
                output = activation_function(Wx_plus_b)
            return output

    # not finished!!!!!!!!!!!!!!!!!!!!!!
    def loss(self, output, y):
        loss = 0
        return loss

    def run_iter(self, verbose=True):
        loss_history = []
        for i in range(len(self.train_data)):
            sess = tf.Session()
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            # every sample go in to train
            node = self.parse(self.train_data[i])

            # add hidden layer with the length of rule
            # for i in range(self.rule_length - 1):  # 调用几次这个函数？计算复合的
            output = self.add_layers(node)

            y = 0  # where is the "y" from?!!!!!!!!!!!!
            loss = self.loss(output, y)  # how to calculate?!!!!!!!!!!!!
            train_step = tf.train.AdagradOptimizer(0.1).minimize(loss)  # learning rate could be halved?
            loss, _ = sess.run([loss, train_step])
            loss_history.append(loss)
            if verbose:
                print("Train step: %d / %d, mean loss: %s" % (i, len(self.train_data), np.mean(loss_history)))
            pass


    def train(self):
        self.load_data()

        for iter in range(self.training_iteration):
            # every iteration is for all data
            self.run_iter()
            pass


def test_RvNN():
    model = RvNN()
    # init(self, dim, rule_length, training_iteration)
    # batch_size
    model.init(100, 2, 1000)


if __name__ == "__main__":
    test_RvNN()
