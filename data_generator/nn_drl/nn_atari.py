# from tensorpack import *
import tensorflow as tf
from tensorpack import argscope, Conv2D, MaxPooling, FullyConnected, PReLU, summary
from tensorpack.train.model_desc import ModelDesc


class Model(ModelDesc):

    def __init__(self, num_actions, image_size):
        self.num_actions = num_actions
        self.state_shape = image_size + (3,)
        self.frame_history = 4

    def inputs(self):
        assert self.num_actions is not None
        return [tf.TensorSpec((None,) + self.state_shape + (self.frame_history,), tf.uint8, 'state'),
                tf.TensorSpec((None,), tf.int64, 'action'),
                tf.TensorSpec((None,), tf.float32, 'futurereward'),
                tf.TensorSpec((None,), tf.float32, 'action_prob'),
                ]

    def _get_NN_prediction(self, state):
        assert state.shape.rank == 5  # Batch, H, W, Channel, History
        state = tf.transpose(state, [0, 1, 2, 4, 3])  # swap channel & history, to be compatible with old models
        image = tf.reshape(state, [-1] + list(self.state_shape[:2]) + [self.state_shape[2] * self.frame_history])

        image = tf.cast(image, tf.float32) / 255.0
        with argscope(Conv2D, activation=tf.nn.relu):
            l = Conv2D('conv0', image, 32, 5)
            l = MaxPooling('pool0', l, 2)
            l = Conv2D('conv1', l, 32, 5)
            l = MaxPooling('pool1', l, 2)
            l = Conv2D('conv2', l, 64, 4)
            l = MaxPooling('pool2', l, 2)
            l = Conv2D('conv3', l, 64, 3)

        l = FullyConnected('fc0', l, 512)
        l = PReLU('prelu', l)
        logits = FullyConnected('fc-pi', l, self.num_actions)    # unnormalized policy
        value = FullyConnected('fc-v', l, 1)
        return logits, value

    def build_graph(self, state, action, futurereward, action_prob):
        logits, value = self._get_NN_prediction(state)
        value = tf.squeeze(value, [1], name='pred_value')  # (B,)
        policy = tf.nn.softmax(logits, name='policy')
        if not self.training:
            return
        log_probs = tf.log(policy + 1e-6)

        log_pi_a_given_s = tf.reduce_sum(
            log_probs * tf.one_hot(action, self.num_actions), 1)
        advantage = tf.subtract(tf.stop_gradient(value), futurereward, name='advantage')

        pi_a_given_s = tf.reduce_sum(policy * tf.one_hot(action, self.num_actions), 1)  # (B,)
        importance = tf.stop_gradient(tf.clip_by_value(pi_a_given_s / (action_prob + 1e-8), 0, 10))

        policy_loss = tf.reduce_sum(log_pi_a_given_s * advantage * importance, name='policy_loss')
        xentropy_loss = tf.reduce_sum(policy * log_probs, name='xentropy_loss')
        value_loss = tf.nn.l2_loss(value - futurereward, name='value_loss')

        pred_reward = tf.reduce_mean(value, name='predict_reward')
        advantage = tf.sqrt(tf.reduce_mean(tf.square(advantage)), name='rms_advantage')
        entropy_beta = tf.get_variable('entropy_beta', shape=[],
                                       initializer=tf.constant_initializer(0.01), trainable=False)
        cost = tf.add_n([policy_loss, xentropy_loss * entropy_beta, value_loss])
        cost = tf.truediv(cost, tf.cast(tf.shape(futurereward)[0], tf.float32), name='cost')
        summary.add_moving_summary(policy_loss, xentropy_loss,
                                   value_loss, pred_reward, advantage,
                                   cost, tf.reduce_mean(importance, name='importance'))
        return cost

    # def optimizer(self):
    #     lr = tf.get_variable('learning_rate', initializer=0.001, trainable=False)
    #     opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)
    #
    #     gradprocs = [MapGradient(lambda grad: tf.clip_by_norm(grad, 0.1 * tf.cast(tf.size(grad), tf.float32))),
    #                  SummaryGradient()]
    #     opt = optimizer.apply_grad_processors(opt, gradprocs)
    #     return opt