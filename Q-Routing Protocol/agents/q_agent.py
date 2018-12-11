import numpy as np
import tensorflow as tf


class NetworkQAgent(object):
    """
    Agent implementing tabular Q-learning for the NetworkSimulatorEnv.
    """

    def __init__(
            self,
            nodes,
            actions,
            node,
            edges_from_node,
            node_to_node,
            absolute_node_edge_tuples,
            destinations,
            n_features,
            learning_rate,
            total_layers,
            layer_size,
            layer_type,
            mean_val,
            std_val,
            constant_val,
            activation_type
    ):

        self.config = {  # cg: reset configuration for each node in the graph
            "init_mean": 0.0,  # Initialize Q values with this mean
            "init_std": 0.0,  # Initialize Q values with this standard deviation
            "learning_rate": 0.7,
            "eps": 0.1,  # Epsilon in epsilon greedy policies
            "discount": 1,
            "n_iter": 10000000}  # Number of iterations
        self.activation_type = activation_type
        self.constant_val = constant_val
        self.destinations = destinations
        self.ep_obs = []
        self.ep_obs2 = []
        self.ep_as = []
        self.ep_rs = []
        self.ep_obs_temp = []
        self.ep_as_temp = []
        self.hist_resources = []
        self.hist_action = []
        self.learning_rate = learning_rate
        self.layer_type = layer_type
        self.links = node_to_node
        self.link_num = absolute_node_edge_tuples
        self.mean_val = mean_val
        self.node = node
        self.n_actions = edges_from_node[self.node]
        self.n_features = n_features
        self.n_links = edges_from_node
        self.num_nodes = nodes
        self.num_actions = actions
        self.total_layers = total_layers
        self.q = []
        self.std_val = std_val

        self.session = tf.Session()
        self._build_net()  # Model
        self.session.run(tf.global_variables_initializer())

        # observations = tf.placeholder(shape=[None, self.n_actions], dtype=tf.float32)
        # actions = tf.placeholder(shape=[None], dtype=tf.float32)
        # rewards = tf.placeholder(shape=[None], dtype=tf.float32)
        # self._build_net_auto(total_layers,layer_size,layer_type,mean_val,std_val,constant_val,activation_type)

    @staticmethod
    def normalize_weights(x):
        """Compute softmax values for each sets of scores in x."""
        """?!--> ????? This is not SoftMax """
        return x / x.sum(axis=0)  # only difference

    @staticmethod
    def next_mini_batch(x_, y_, z_, batch_size):
        """""""?!--> what are these x, y, and z, representative of?"""
        """Create a vector with batch_size quantity of random integers; generate a mini-batch therefrom???."""
        permutation = np.random.permutation(x_.shape[0])
        permutation = permutation[:batch_size]
        x_batch = x_[permutation, :]
        y_batch = y_[permutation]
        z_batch = z_[permutation]
        return x_batch, y_batch, z_batch

    # called in initializer
    def _build_net(self):
        """
        tf.name_scope is a context manager for defining Python operations
        tf.placeholder returns a `Tensor` that may be used as a handle for feeding a value, but not evaluated directly.
        """
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
            self.tf_actions_value = \
                tf.placeholder(
                    dtype=tf.float32,
                    shape=[None, ],
                    name="actions_value"
                )
        # fc1
        # https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/7_Policy_gradient_softmax/RL_brain.py

        """ 
        tf.layers.dense - 
        description: 
            Functional interface for the densely-connected layer that implements the operation: 
            activation(inputs * kernel + bias) 
            where activation is the activation function passed as the activation argument (if not None), kernel is a 
            weights matrix created by the layer, and bias is a bias vector created by the layer (only if use_bias is 
            True).
        Inputs: 
            inputs: Tensor input.
            units: Integer or Long, dimensionality of the output space.
            activation: Activation function (callable). Set it to None to maintain a linear activation.
            use_bias: Boolean, whether the layer uses a bias.
            kernel_initializer: Initializer function for the weight matrix. If None (default), weights are 
                                initialized using the default initializer used by tf.get_variable.
            bias_initializer: Initializer function for the bias.
            kernel_regularizer: Regularizer function for the weight matrix.
            bias_regularizer: Regularizer function for the bias.
            activity_regularizer: Regularizer function for the output.
            kernel_constraint: An optional projection function to be applied to the kernel after being updated by an
                               Optimizer (e.g. used to implement norm constraints or value constraints for layer 
                               weights). The function must take as input the unprojected variable and must return 
                               the projected variable (which must have the same shape). Constraints are not safe to 
                               use when doing asynchronous distributed training.
            bias_constraint: An optional projection function to be applied to the bias after being updated by an 
                             Optimizer.
            trainable: Boolean, if True also add variables to the graph collection GraphKeys.TRAINABLE_VARIABLES 
                       (see tf.Variable).
            name: String, the name of the layer.
            reuse: Boolean, whether to reuse the weights of a previous layer by the same name.
        Output: 
            tensor the same shape as inputs except the last dimension is of size units
        """

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
                    input_tensor=self.neg_log_prob * self.tf_actions_value,
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
                    self.tf_actions_value: z_batch,  # shape=[None, ]
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
