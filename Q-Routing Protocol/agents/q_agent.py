import numpy as np
import tensorflow as tf


class NetworkTabularQAgent(object):
    """
    Agent implementing tabular Q-learning for the NetworkSimulatorEnv.
    """

    def __init__(
            self,
            num_nodes,
            num_actions,
            node,
            n_links,
            links,
            link_num,
            destinations,
            n_features,
            learning_rate,
            num_layers,
            layer_size,
            layer_type,
            mean_val,
            std_val,
            constant_val,
            activation_type
    ):

        # cg: reset configuration for each node in the graph
        self.config = {
            "init_mean": 0.0,  # Initialize Q values with this mean
            "init_std": 0.0,  # Initialize Q values with this standard deviation
            "learning_rate": 0.7,
            "eps": 0.1,  # Epsilon in epsilon greedy policies
            "discount": 1,
            "n_iter": 10000000}  # Number of iterations
        self.num_nodes = num_nodes
        self.num_actions = num_actions
        self.node = node
        self.n_links = n_links
        self.links = links
        self.link_num = link_num
        self.destinations = destinations
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        self.layer_type = layer_type
        self.mean_val = mean_val
        self.std_val = std_val
        self.constant_val = constant_val
        self.activation_type = activation_type
        self.n_actions = n_links[self.node]
        self.ep_obs = []
        self.ep_obs2 = []
        self.ep_as = []
        self.ep_rs = []
        self.ep_obs_temp = []
        self.ep_as_temp = []
        self.hist_resources = []
        self.hist_action = []
        self.q = []

        self.session = tf.Session()
        self._build_net()  # Model
        self.session.run(tf.global_variables_initializer())

        # observations = tf.placeholder(shape=[None, self.n_actions], dtype=tf.float32)
        # actions = tf.placeholder(shape=[None], dtype=tf.float32)
        # rewards = tf.placeholder(shape=[None], dtype=tf.float32)
        # self._build_net_auto(num_layers,layer_size,layer_type,mean_val,std_val,constant_val,activation_type)

    @staticmethod
    def normalize_weights(x):
        """Compute softmax values for each sets of scores in x."""
        return x / x.sum(axis=0)  # only difference

    @staticmethod
    def next_mini_batch(x_, y_, z_, batch_size):
        """Create a vector with batch_size quantity of random integers; generate a mini-batch therefrom."""
        perm = np.random.permutation(x_.shape[0])
        perm = perm[:batch_size]
        x_batch = x_[perm, :]
        y_batch = y_[perm]
        z_batch = z_[perm]
        return x_batch, y_batch, z_batch

    # called in initializer
    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_observations = \
                tf.placeholder(
                    dtype=tf.float32,
                    shape=[None, self.n_features],
                    name="observations"
                )
            self.tf_observations_2 = \
                tf.placeholder(
                    dtype=tf.float32,
                    shape=[None, self.n_features],
                    name="observations"
                )
            self.tf_action_number = \
                tf.placeholder(
                    dtype=tf.int32,
                    shape=[None, ],
                    name="actions_num"
                )
            self.tf_vt = \
                tf.placeholder(
                    dtype=tf.float32,
                    shape=[None, ],
                    name="actions_value"
                )
        # fc1
        # https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/7_Policy_gradient_softmax/RL_brain.py
        self.layer = tf.layers.dense(
            inputs=self.tf_observations,
            units=50,
            activation=None,  # tf.nn.relu,  # tanh activation
            use_bias=True,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=.1),
            bias_initializer=tf.constant_initializer(1),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            trainable=True,
            name=None,
            reuse=None
        )
        layer2 = tf.layers.dense(
            inputs=self.layer,
            units=25,
            activation=tf.nn.relu,  # tf.nn.relu,  # tanh activation
            use_bias=True,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=.1),
            bias_initializer=tf.constant_initializer(1),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            trainable=True,
            name=None,
            reuse=None
        )
        layer3 = tf.layers.dense(
            inputs=layer2,
            units=15,
            activation=tf.nn.sigmoid,  # tf.nn.relu,  # tanh activation
            use_bias=True,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=.1),
            bias_initializer=tf.constant_initializer(1),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            trainable=True,
            name=None,
            reuse=None
        )
        # fc2
        self.all_act = tf.layers.dense(
            inputs=layer3,
            units=self.n_actions,
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=.1),
            bias_initializer=tf.constant_initializer(1),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            trainable=True,
            name=None,
            reuse=None
        )
        # use softmax to convert to probability
        self.action_probabilities = tf.nn.softmax(logits=self.all_act, axis=None, name=None, )

        with tf.name_scope('loss'):
            one_hot_tensor = \
                tf.one_hot(
                    indices=self.tf_action_number,
                    depth=self.n_actions,
                    on_value=None,
                    off_value=None,
                    axis=None,
                    dtype=None,
                    name=None
                )
            neg_logarithm_action_probabilities = \
                -tf.log(
                    x=self.action_probabilities,
                    name=None
                )
            self.neg_log_prob = \
                tf.reduce_sum(
                    input_tensor=neg_logarithm_action_probabilities * one_hot_tensor,
                    axis=1,
                    keepdims=None,
                    name=None,
                    reduction_indices=None
                )
            # Reward guided loss
            self.loss = \
                tf.reduce_mean(
                    input_tensor=self.neg_log_prob * self.tf_vt,
                    axis=None,
                    keepdims=None,
                    name=None,
                    reduction_indices=None
                )
            print("Why is there a print command here, and why print help?")

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def choose_action(self, observation, valid):
        prob_weights = \
            self.session.run(
                fetches=self.action_probabilities,
                feed_dict={self.tf_observations: observation},
                options=None,
                run_metadata=None
            )
        valid_weights = prob_weights * valid
        valid_prob = self.normalize_weights(valid_weights[0])
        action = \
            np.random.choice(
                a=range(prob_weights.shape[1]),
                size=None,
                replace=True,
                p=valid_prob.ravel()
            )
        return action

    def choose_action2(self, observation):
        prob_weights = \
            self.session.run(
                fetches=self.action_probabilities,
                feed_dict={self.tf_observations: observation},
                options=None,
                run_metadata=None
            )
        action = \
            np.random.choice(
                a=range(prob_weights.shape[1]),
                size=None,
                replace=True,
                p=prob_weights.ravel()
            )
        return action

    def store_transition(self, state, action, reward):
        self.ep_obs.append(state)
        self.ep_as.append(action)
        self.ep_rs.append(reward)

    def store_transition_temp(self, state, action):
        self.ep_obs_temp.append(state)
        self.ep_as_temp.append(action)

    def store_transition_episode(self, reward):
        # ?! --> is it duration though?
        ep_as_temp = len(self.ep_as_temp)
        for i in range(0, ep_as_temp):
            self.store_transition(
                self.ep_obs_temp[i],
                self.ep_as_temp[i],
                reward
            )

    def learn5(self, iteration):
        ep_obs = len(self.ep_obs)
        self.ep_obs2 = np.array(self.ep_obs).reshape(ep_obs, self.n_features)
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        x_batch, y_batch, z_batch = \
            self.next_mini_batch(
                np.vstack(self.ep_obs2),
                np.array(self.ep_as),
                np.array(discounted_ep_rs_norm),
                ep_obs
            )
        _, loss, log_probabilities, act_val = \
            self.session.run(
                fetches=[self.train_op, self.loss, self.neg_log_prob, self.all_act],
                feed_dict={
                    self.tf_observations: x_batch,  # shape=[None, n_obs]
                    self.tf_action_number: y_batch,  # shape=[None, ]
                    self.tf_vt: z_batch,  # shape=[None, ]
                },
                options=None,
                run_metadata=None
            )
        if iteration % 1 == 0:
            self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # empty episode data

    def _discount_and_norm_rewards(self):
        self.gamma, running_add = 0, 0
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

    def act_nn2(self, resources_edges, resources_bbu):
        edge_bbu_sum = resources_edges + resources_bbu
        obs = np.array(edge_bbu_sum).reshape(1, self.n_features)
        action = self.choose_action2(obs)
        self.store_transition_temp(edge_bbu_sum, action)
        next_node = self.links[self.node][action]
        # l_num = self.link_num[self.node][action]
        if resources_edges[self.link_num[self.node][action]] == 0:
            action = -1
        elif next_node in self.destinations:
            if resources_bbu[self.destinations.index(next_node)] == 0:
                action = -1
        return action
