import numpy as np
import tensorflow as tf

# TODO: Change from policy gradient to batch actor-critic - make nn's output
# TODO: a value, then for loss, make loss the loggrad*(rew+V(s_t+1)-V(s_t)). Use nn to get s_t+1
# TODO: http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_5_actor_critic_pdf.pdf slide 21


class NetworkQAgent(object):
    """Agent implementing Q-learning for the NetworkSimulatorEnv."""

    def __init__(
            self,
            nodes,
            node,
            edges_from_node,
            node_to_node,
            absolute_node_edge_tuples,
            destinations,
            n_features,
            learning_rate,
            total_layers,
            layer_type,
            mean_val,
            std_val,
            constant_val,
    ):

        self.config = {  # cg: reset configuration for each node in the graph
            "init_mean": 0.0,  # Initialize Q values with this mean
            "init_std": 0.0,  # Initialize Q values with this standard deviation
            "learning_rate": 0.7,
            "eps": 0.1,  # Epsilon in epsilon greedy policies
            "discount": 1,
            "n_iter": 1000}  # Number of iterations
        self.constant_val = constant_val
        self.destinations = destinations
        self.episode_observation = []
        self.episode_observation2 = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_observation_temp = []
        self.episode_actions_temp = []
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
        self.total_layers = total_layers
        self.q = []
        self.std_val = std_val
        self.val_approx = np.random.rand()


# PANDASAURUS REX

        self.session = tf.Session()
        self._build_net()  # Model
        self.session.run(tf.global_variables_initializer())

    @staticmethod
    def normalize_weights(x):
        """Compute SoftMax values for each sets of scores in x"""
        """?!--> ????? This is not SoftMax """
        return x / x.sum(axis=0)  # only difference

    @staticmethod
    def next_mini_batch(x_, y_, z_, batch_size):
        """Create a vector with batch_size quantity of random integers dictating the mini_batch size.
        Recall that mini-batches is the subset of the larger set of training instances. The algorithm must observe
        mini-batch size of training instances before updating parameters instead of observing all training instances"""
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
                    name=None
                )

        # https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/7_Policy_gradient_softmax/RL_brain.py

        """Pass in the environment state and agent actions to the neural net"""
        # --> Forward Fully Connected Layer 1
        self.layer = \
            tf.layers.dense(
                inputs=self.tf_observations,
                units=50,
                activation=None,  # tf.nn.relu,  # tanh activation
                use_bias=True,
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=.1),
                bias_initializer=tf.constant_initializer(1),
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                trainable=True,
                name=None,
                reuse=None
            )
        # --> Forward Fully Connected Layer 2
        layer2 = \
            tf.layers.dense(
                inputs=self.layer,
                units=25,
                activation=tf.nn.relu,  # tf.nn.relu,  # tanh activation
                use_bias=True,
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=.1),
                bias_initializer=tf.constant_initializer(1),
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                trainable=True,
                name=None,
                reuse=None
            )
        # --> Forward Fully Connected Layer 3
        layer3 = \
            tf.layers.dense(
                inputs=layer2,
                units=15,
                activation=tf.nn.sigmoid,  # tf.nn.relu,  # tanh activation
                use_bias=True,
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=.1),
                bias_initializer=tf.constant_initializer(1),
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                trainable=True,
                name=None,
                reuse=None
            )
        # --> Forward Fully Connected Layer 4
        self.all_act = \
            tf.layers.dense(
                inputs=layer3,
                units=self.n_actions,
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=.1),
                bias_initializer=tf.constant_initializer(1),
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                trainable=True,
                name=None,
                reuse=None
            )

        """Convert neural network output to probability distribution"""
        self.action_probabilities = tf.nn.softmax(logits=self.all_act, name="action_probabilities")

        with tf.name_scope('loss'):
            one_hot_tensor = \
                tf.one_hot(
                    indices=self.tf_action_number,
                    depth=self.n_actions,
                    on_value=None,
                    off_value=None,
                    axis=None,
                    dtype=None,
                    name="one_hot_tensor"
                )

            self.neg_log_prob = \
                tf.reduce_sum(
                    input_tensor=-tf.log(x=self.action_probabilities, name="neg_log_act_prob") * one_hot_tensor,
                    axis=1,
                    name="reduce_sum",
                    reduction_indices=None
                )

            # Reward guided loss
            self.loss = \
                tf.reduce_mean(
                    input_tensor=self.neg_log_prob * (self.tf_vt - self.val_approx),
                    axis=None,
                    name="reduce_mean",
                    reduction_indices=None
                )

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
        self.episode_observation.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)

    def store_transition_temp(self, state, action):
        self.episode_observation_temp.append(state)
        self.episode_actions_temp.append(action)

    def store_transition_episode(self, reward):
        ep_as_temp = len(self.episode_actions_temp)
        for i in range(0, ep_as_temp):
            self.store_transition(
                self.episode_observation_temp[i],
                self.episode_actions_temp[i],
                reward
            )

    def learn5(self, iteration, val_approx):
        self.val_approx = val_approx
        # print(self.val_approx)
        len_obs = len(self.episode_observation)
        self.episode_observation2 = np.array(self.episode_observation).reshape(len_obs, self.n_features)
        discounted_episode_rewards_norm = self._discount_and_norm_rewards()
        # print('discounted_episode_rewards_norm =',discounted_episode_rewards_norm)
        x_batch, y_batch, z_batch = \
            self.next_mini_batch(
                self.episode_observation2,
                np.array(self.episode_actions),
                np.array(discounted_episode_rewards_norm),
                # len_obs
                len_obs
            )
        # print('from policy_nn: x_batch.shape =', x_batch.shape)
        # print('from policy_nn: z_batch.shape =', z_batch.shape)
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
            # empty episode input_data
            self.episode_observation, self.episode_actions, self.episode_rewards = [], [], []
            self.episode_observation_temp,self.episode_actions_temp = [],[]

    def _discount_and_norm_rewards(self):
        self.gamma, running_add = 0, 0
        discounted_episode_rewards = np.zeros_like(self.episode_rewards)
        for t in reversed(range(0, len(self.episode_rewards))):
            running_add = running_add * self.gamma + self.episode_rewards[t]
            discounted_episode_rewards[t] = running_add
        discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        discounted_episode_rewards /= np.std(discounted_episode_rewards)
        return discounted_episode_rewards

    def act_nn2(self, resources_edges, resources_bbu):
        edge_resources = resources_edges + resources_bbu
        obs = np.array(edge_resources).reshape(1, self.n_features)
        action = self.choose_action2(obs)
        self.store_transition_temp(edge_resources, action)
        next_node = self.links[self.node][action]
        # l_num = self.link_num[self.node][action]
        if resources_edges[self.link_num[self.node][action]] == 0:
            action = -1
        elif next_node in self.destinations:
            if resources_bbu[self.destinations.index(next_node)] == 0:
                action = -1
        return action


class NetworkValAgent(object):
    """Agent to evaluate state of NetworkSimulatorEnv"""
    def __init__(
            self,
            nodes,
            node,
            edges_from_node,
            node_to_node,
            absolute_node_edge_tuples,
            destinations,
            n_features,
            learning_rate,
            total_layers,
            layer_type,
            mean_val,
            std_val,
            constant_val,
    ):

        self.config = {  # cg: reset configuration for each node in the graph
            "init_mean": 0.0,  # Initialize Q values with this mean
            "init_std": 0.0,  # Initialize Q values with this standard deviation
            "learning_rate": 0.7,
            "eps": 0.1,  # Epsilon in epsilon greedy policies
            "discount": 1,
            "n_iter": 1000}  # Number of iterations
        self.constant_val = constant_val
        self.destinations = destinations
        self.episode_observation = []
        self.episode_observation2 = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_observation_temp = []
        self.episode_actions_temp = []
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
        self.total_layers = total_layers
        self.q = []
        self.std_val = std_val

        self.session = tf.Session()
        self._build_value_net()  # Model
        self.session.run(tf.global_variables_initializer())

    @staticmethod
    def normalize_weights(x):
        """Compute softmax values for each sets of scores in x."""
        """?!--> ????? This is not SoftMax """
        return x / x.sum(axis=0)  # only difference

    @staticmethod
    def next_mini_batch(x_, z_, batch_size):
        """Create a vector with batch_size quantity of random integers dictating the mini_batch size.
        Recall that mini-batches is the subset of the larger set of training instances. The algorithm must observe
        mini-batch size of training instances before updating parameters instead of observing all training instances"""
        permutation = np.random.permutation(x_.shape[0])
        permutation = permutation[:batch_size]
        x_batch = x_[permutation, :]
        z_batch = z_[permutation]
        return x_batch, z_batch

    def _build_value_net(self):

        with tf.name_scope('inputs'):
            self.tf_observations = \
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
                    name=None
                )

        # https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/7_Policy_gradient_softmax/RL_brain.py

        # --> Forward Connected Layer 1
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
            trainable=True,
            name=None,
            reuse=None
        )
        # --> Forward Connected Layer 2
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
            trainable=True,
            name=None,
            reuse=None
        )
        # --> Forward Connected Layer 3
        layer3 = tf.layers.dense(
            inputs=layer2,
            units=15,
            activation=tf.nn.relu,  # tf.nn.relu,  # tanh activation
            use_bias=True,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=.1),
            bias_initializer=tf.constant_initializer(1),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            trainable=True,
            name=None,
            reuse=None
        )
        # --> Forward Connected Layer 4
        self.val_approx = tf.layers.dense(
            inputs=layer3,
            units=1,
            activation=None,  # tf.nn.relu, None tf.nn.sigmoid
            use_bias=True,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=.3),
            bias_initializer=tf.constant_initializer(0),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            trainable=True,
            name=None,
            reuse=None
        )

        with tf.name_scope('value_loss'):
            # Reward guided value_loss
            self.value_loss = tf.nn.l2_loss(self.val_approx - self.tf_vt)

        with tf.name_scope('train'):
            self.train_value_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.value_loss)

    def store_transition(self, state, reward):
        self.episode_observation.append(state)
        self.episode_rewards.append(reward)

    def store_transition_temp(self, state):
        self.episode_observation_temp.append(state)

    def store_transition_episode(self, reward):
        ep_as_temp = len(self.episode_observation_temp)
        for i in range(0, ep_as_temp):
            self.store_transition(
                self.episode_observation_temp[i],
                reward
            )

    def eval_state(self, observation):
        val = self.session.run(
            fetches=self.val_approx,
            feed_dict={self.tf_observations: observation},
            options=None,
            run_metadata=None
        )
        return val

    def learn_val(self, iteration):
        len_obs = len(self.episode_observation)
        self.episode_observation2 = np.array(self.episode_observation).reshape(len_obs, self.n_features)
        discounted_episode_rewards_norm = self._discount_and_norm_rewards()
        x_batch, z_batch = \
            self.next_mini_batch(
                self.episode_observation2,
                np.array(discounted_episode_rewards_norm),
                len_obs
            )
        # print('from val_nn: x_batch.shape =',x_batch.shape)
        # print('from val_nn: z_batch.shape =',z_batch.shape)
        _, loss = \
            self.session.run(
                fetches=[self.train_value_op, self.value_loss],
                feed_dict={
                    self.tf_observations: x_batch,  # shape=[None, n_obs]
                    self.tf_vt: z_batch,  # shape=[None, ]
                },
                options=None,
                run_metadata=None
            )
        if iteration % 1 == 0:
            # empty episode input_data
            self.episode_observation, self.episode_actions, self.episode_rewards = [], [], []
            self.episode_observation_temp = []

    def _discount_and_norm_rewards(self):
        self.gamma, running_add = 0, 0
        discounted_episode_rewards = np.zeros_like(self.episode_rewards)
        for t in reversed(range(0, len(self.episode_rewards))):
            running_add = running_add * self.gamma + self.episode_rewards[t]
            discounted_episode_rewards[t] = running_add
        discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        discounted_episode_rewards /= np.std(discounted_episode_rewards)
        return discounted_episode_rewards

    def eval_nn(self, resources_edges, resources_bbu):
        edge_resources = resources_edges + resources_bbu
        obs = np.array(edge_resources).reshape(1, self.n_features)
        val = self.eval_state(obs)
        self.store_transition_temp(edge_resources)
        return val[0][0]

