import numpy as np
import tensorflow as tf

<<<<<<< HEAD
class networkTabularQAgent(object):
  """
  Agent implementing tabular Q-learning for the NetworkSimulatorEnv.
  """

  def __init__(
      self, 
      num_nodes, 
      num_actions, 
      node, 
      nlinks, 
      links, 
      link_num, 
      dests, 
      n_features, 
      learning_rate, 
      num_layers, 
      layer_size, 
      layer_type, 
      mean_val, 
      std_val, 
      constant_val, 
      activation_type):

    # cg: reset configuration for each node in the graph by initializing Q-values' means, standard deviations, 
    # learning rates, greediness, and discount factors
    self.config = { 
      "init_mean": 0.0, 
      "init_std": 0.0, 
      "learning_rate" : 0.7, 
      "eps": 0.1, 
      "discount": 1, 
      "n_iter": 10000000
      } 
    # self.q = np.zeros((num_nodes, num_nodes, num_actions))

    self.hist_resources, self.hist_action = ([],)*2
    # self.hist_resources, self.hist_action = [], []
    self.ep_obs, self.ep_as, self.ep_rs = ([],)*3
    # self.ep_obs, self.ep_as, self.ep_rs = [], [], []
    self.ep_obs_temp, self.ep_as_temp = ([],)*2
    # self.ep_obs_temp, self.ep_as_temp = [], []
    self.n = node
    self.links = links
    self.link_num = link_num
    self.dests = dests
    # self.nlinks = nlinks
    self.n_actions = nlinks[self.n]
    self.n_features = n_features
    self.lr = learning_rate 

    self.sess = tf.Session()

    observations = tf.placeholder(
                                  shape = [None, self.n_actions], 
                                  dtype = tf.float32
                                  )
    actions = tf.placeholder(
                            shape = [None], 
                            dtype = tf.float32
                            )
    rewards = tf.placeholder(
                            shape = [None], 
                            dtype = tf.float32
                            )
    # actions, rewards = (tf.placeholder(shape = [None], dtype = tf.float32),)*2

    # model
    self._build_net()
    #self._build_net_auto(num_layers, layer_size, layer_type, mean_val, std_val, constant_val, activation_type)
    self.sess.run(tf.global_variables_initializer())

  def normalize_weights(self, x):
    """Compute softmax values for each sets of scores in x."""
    return x / x.sum(axis = 0) # only difference

  def next_minibatch(self, X_, Y_, Z_, batch_size):
    # Create a vector with batch_size random integers
    perm = np.random.permutation(X_.shape[0])
    perm = perm[:batch_size]
    # Generate the minibatch
    X_batch = X_[perm, :]
    Y_batch = Y_[perm]
    Z_batch = Z_[perm]
    # Return the images and the labels
    return X_batch, Y_batch, Z_batch

  def _build_net(self):
    with tf.name_scope('inputs'):
      self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name = "observations")
      self.tf_obs2 = tf.placeholder(tf.float32, [None, self.n_features], name = "observations")
      self.tf_acts = tf.placeholder(tf.int32, [None, ], name = "actions_num")
      self.tf_vt = tf.placeholder(tf.float32, [None, ], name = "actions_value")
    # fc1
    #https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/7_Policy_gradient_softmax/RL_brain.py
    self.layer = tf.layers.dense(
      inputs = self.tf_obs, 
      units = 50, 
      activation = None,  # tf.nn.relu, # tanh activation
      kernel_initializer = tf.random_normal_initializer(mean = 0, stddev = .1), 
      bias_initializer = tf.constant_initializer(1)
    )

    layer2 = tf.layers.dense(
      inputs = self.layer, 
      units = 25, 
      activation = tf.nn.relu,  # tf.nn.relu, # tanh activation
      kernel_initializer = tf.random_normal_initializer(mean = 0, stddev = .1), 
      bias_initializer = tf.constant_initializer(1)
    )

    layer3 = tf.layers.dense(
      inputs = layer2, 
      units = 15, 
      activation = tf.nn.sigmoid,  # tf.nn.relu, # tanh activation
      kernel_initializer = tf.random_normal_initializer(mean = 0, stddev = .1), 
      bias_initializer = tf.constant_initializer(1)
    )

    # fc2
    self.all_act = tf.layers.dense(
      inputs = layer3, 
      units = self.n_actions, 
      activation = tf.nn.relu, 
      kernel_initializer = tf.random_normal_initializer(mean = 0, stddev = .1), 
      bias_initializer = tf.constant_initializer(1)
    )

    self.all_act_prob = tf.nn.softmax(self.all_act) # use softmax to convert to probability

    with tf.name_scope('loss'):
      self.neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob) * tf.one_hot(self.tf_acts, self.n_actions), axis = 1)
      # reward guided loss
      self.loss = tf.reduce_mean(self.neg_log_prob * self.tf_vt)
      print("help")
      
    with tf.name_scope('train'):
      self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

  def _build_net_auto(self, num_layers, layer_size, layer_type, mean_val, std_val, constant_val, activation_type):
    with tf.name_scope('inputs'):
      self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name = "observations")
      self.tf_obs2 = tf.placeholder(tf.float32, [None, self.n_features], name = "observations")
      self.tf_acts = tf.placeholder(tf.int32, [None, ], name = "actions_num")
      self.tf_vt = tf.placeholder(tf.float32, [None, ], name = "actions_value")
    
    # fc1
    #https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/7_Policy_gradient_softmax/RL_brain.py
    act = [None, tf.nn.relu, tf.nn.sigmoid]
    for i in range(num_layers):
      if layer_type[i] == 'first':
        temp_layer = tf.layers.dense(
          inputs = self.tf_obs, 
          units = layer_size[i], 
          activation = act[activation_type[i]], # activation_type[i], 
          kernel_initializer = tf.random_normal_initializer(mean = mean_val[i], stddev = std_val[i]), 
          bias_initializer = tf.constant_initializer(constant_val[i])
        )
      if layer_type[i] == 'middle':
        temp_layer = tf.layers.dense(
          inputs = temp_layer_in, 
          units = layer_size[i], 
          activation = act[activation_type[i]], 
          kernel_initializer = tf.random_normal_initializer(mean = mean_val[i], stddev = std_val[i]), 
          bias_initializer = tf.constant_initializer(constant_val[i])
=======

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
        self.observation_length = []
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
        """?!--> what are these x, y, and z, representative of?"""
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
>>>>>>> 65476e1cd7ab8479fad251f0a2a3c728036b9038
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

        # --> Forward Connected Layer 4
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
            trainable=True,
            name=None,
            reuse=None
        )


        # use SoftMax to convert to probability
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
            neg_logarithm_action_probabilities = \
                -tf.log(
                    x=self.action_probabilities,
                    name="negative_log_action_probabilities"
                )

            self.neg_log_prob = \
                tf.reduce_sum(
                    input_tensor=neg_logarithm_action_probabilities * one_hot_tensor,
                    axis=1,
                    name="reduce_sum",
                    reduction_indices=None
                )
            # Reward guided loss
            self.loss = \
                tf.reduce_mean(
                    input_tensor=self.neg_log_prob * self.tf_vt,
                    axis=None,
                    name="reduce_mean",
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
        self.observation_length.append(state)
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

    def learn5(self, iteration):
        episode_observation = len(self.observation_length)
        self.episode_observation2 = np.array(self.observation_length).reshape(episode_observation, self.n_features)
        discounted_episode_rewards_norm = self._discount_and_norm_rewards()
        # print('self.episode_observation2.shape =', self.episode_observation2.shape)
        x_batch, y_batch, z_batch = \
            self.next_mini_batch(
                self.episode_observation2,
                np.array(self.episode_actions),
                np.array(discounted_episode_rewards_norm),
                episode_observation
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
            self.observation_length, self.episode_actions, self.episode_rewards = [], [], []  # empty episode input_data

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
