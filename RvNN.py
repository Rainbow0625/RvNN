import tensorflow as tf
import numpy as np


class TreeNode:
    def __init__(self):
        self.index = -1
        self.emb = None  # the embedding of the predicate
        self.parent = None
        self.left = None
        self.right = None
        self.isLeaf = False


class RvNN(object):
    def __int__(self, dim, rule_length, training_iteration, learning_rate, l2):
        self.dimension = dim
        self.rule_length = rule_length
        self.training_iteration = training_iteration
        self.learning_rate = learning_rate
        self.l2 = l2
        self.model_name = 'rnn_embed=%d_l2=%f_lr=%f.weights' % (dim, l2, learning_rate)

    # not finished!!!!!!!!!!!!!!!!!!!!!!
    def load_data(self):
        self.train_data = []
        self.test_data = []

    def add_layer(self, x_c1, x_c2, activation_function=tf.tanh):  # tf.sigmoid 待定
        regularization = tf.contrib.layers.l2_regularizer(0.0001)  # L2 regularizion
        with tf.variable_scope('Composition', initializer=tf.random_normal_initializer,
                               regularizer=regularization):
            weight = tf.get_variable('weight', shape=[self.dimension, self.dimension * 2])
            bias = tf.get_variable('bias', shape=[self.dimension, 1])
        # define the placeholder for inputs to network
        # x_c1 = tf.placeholder(tf.float32, [None, self.dimension])
        # x_c2 = tf.placeholder(tf.float32, [None, self.dimension])
        x = tf.concat([x_c1, x_c2], 0)
        Wx_plus_b = tf.matmul(weight, x) + bias
        if activation_function is None:
            output = Wx_plus_b
        else:
            output = activation_function(Wx_plus_b)
        return output

    def parse_to_tree(self, tokens):
        input_list = [item for item in tokens]
        node_list = []
        for p in input_list:
            # p[0]: index   p[1]: embedding
            node = TreeNode()  # data includes embedding
            node.index = p[0]
            node.emb = p[1]
            node.isLeaf = True
            node_list.append(node)
        p = TreeNode()
        p.isLeaf = False
        p.left = node_list[0]
        node_list[0].parent = p
        p.right = node_list[1]
        node_list[1].parent = p
        p.emb = self.add_layer(node_list[0].emb, node_list[1].emb)
        for i in range(self.rule_length-1):
            p_parent = TreeNode()
            p_parent.isLeaf = False
            p_parent.left = p
            p.parent = p_parent
            p_parent.right = node_list[i+2]
            node_list[i+2].parent = p_parent
            p_parent.emb = self.add_layer(p.emb, node_list[i+2].emb)
            p = p_parent
        return p

    def loss(self, output, y):
        with tf.variable_scope('Composition', reuse=True):
            weight = tf.get_variable("weight")
        # L2 Regularization + Mean Squared Error
        mse = tf.reduce_sum(tf.square(y - output))
        # mse = tf.losses.mean_squared_error(y, output)
        loss = tf.nn.l2_loss(weight) * self.l2 + mse
        return loss

    # not finished!!!!!!!!!!!!!!!!!!!!!!
    def predict(self, data):
        results = []
        with tf.Graph().as_default(), tf.Session() as sess:
            for i in range(len(data)):
                saver = tf.train.Saver()
                saver.restore(sess, './weights/%s.temp' % self.model_name)
                root_node = self.parse_to_tree((data[i]))
                y = 0  # where is the "y" from?!!!!!!!!!!!!
                root_node = sess.run(root_node)
                prediction = root_node.emb
                results.append(prediction)
        # how to calculate the accuracy!!!!!!!!! results compare to y?
        accuracy = 0.0
        return accuracy

    def run_iter(self, new_model = False, verbose=True):
        loss_history = []
        with tf.Graph().as_default(), tf.Session() as sess:
            for i in range(len(self.train_data)):
                if new_model:
                    init_op = tf.global_variables_initializer()
                    sess.run(init_op)
                else:
                    saver = tf.train.Saver()
                    saver.restore(sess, './weights/%s.temp' % self.model_name)
                # every sample go in to train
                root_node = self.parse_to_tree(self.train_data[i])
                # minimize the loss
                output = root_node.emb
                y = 0  # where is the "y" from?!!!!!!!!!!!!
                loss = self.loss(output, y)
                # learning rate could be halved?
                train_step = tf.train.AdagradOptimizer(self.learning_rate).minimize(loss)
                loss, _ = sess.run([loss, train_step])
                loss_history.append(loss)
                if verbose:
                    print("   Train step: %d / %d, mean loss: %s" % (i, len(self.train_data), np.mean(loss_history)))
            saver = tf.train.Saver()
            saver.save(sess, './weights/%s.temp' % self.model_name)
        # after training stage, we should see the train result of the model
        train_accuracy = self.predict(self.train_data)
        print("   accuracy: %f" % train_accuracy)
        return train_accuracy

    def train(self):
        print("Training begins.")
        for step in range(self.training_iteration):
            # every iteration is for all data
            print(" Iteration %d: \n" % step)
            if iter == 0:
                train_accuracy = self.run_iter(new_model=True)
            else:
                train_accuracy = self.run_iter()
        print("Training ends.")


def test_RvNN():
    Dimension = 100  # 50 in paper
    Rule_Length = 2
    Training_Iteration = 100
    Learning_Rate = 0.1
    L2 = 0.0001

    model = RvNN()
    model.__int__(Dimension, Rule_Length, Training_Iteration, Learning_Rate, L2)
    model.load_data()
    model.train()


if __name__ == "__main__":
    test_RvNN()
