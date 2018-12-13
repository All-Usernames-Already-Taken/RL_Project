import numpy as np
import tensorflow as tf


class NetworkQAgent(object):
    """
    Agent implementing Q-learning for the NetworkSimulatorEnv.
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
            "n_iter": 1000}  # Number of iterations
        self.activation_type = activation_type
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
        self.num_actions = actions
        self.total_layers = total_layers
        self.q = []
        self.std_val = std_val

        try:
            with tf.device('/gpu:0'):
                self._build_net()  # Model
                self.session = tf.Session()
                self.session.run(tf.global_variables_initializer())
        except:
            pass
        else:
            self._build_net()  # Model
            self.session = tf.Session()
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
                    name="observations2"
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

        """ 
        tf.layers.dense - 
            Description: 
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

        """
        tf.nn.softmax - 
            Aliases:
                tf.math.softmax
                tf.nn.softmax
            Description:
                Computes softmax activations. 
                This function performs the equivalent of softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)
            Args:
                logits: A non-empty Tensor. Must be one of the following types: half, float32, float64.
                axis: The dimension softmax would be performed on. The default is -1 which indicates the last dimension.
                name: A name for the operation (optional).
                dim: Deprecated alias for axis.
            Returns:
                A Tensor. Has the same type and shape as logits.
        
            Raises:
                InvalidArgumentError: if logits is empty or axis is beyond the last dimension of logits.
        """

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

            """
            tf.math.reduce_sum
                Aliases:
                    tf.math.reduce_sum
                    tf.reduce_sum
                Description:       
                    Computes the sum of elements across dimensions of a tensor. (deprecated arguments)
                    Reduces input_tensor along the dimensions given in axis. Unless keepdims is true, the rank of the 
                    tensor is reduced by 1 for each entry in axis. If keepdims is true, the reduced dimensions are 
                    retained with length 1.
                    If axis is None, all dimensions are reduced, and a tensor with a single element is returned.
                Args:
                    input_tensor: The tensor to reduce. Should have numeric type.
                    axis: The dimensions to reduce. If None (the default), reduces all dimensions. Must be in the range
                          [-rank(input_tensor), rank(input_tensor)).
                    keepdims: If true, retains reduced dimensions with length 1.
                    name: A name for the operation (optional).
                    reduction_indices: The old (deprecated) name for axis.
                    keep_dims: Deprecated alias for keepdims.
                Returns:
                    The reduced tensor, of the same dtype as the input_tensor.
            """

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

    """
    tf.session.run
        Descriptions:
            Runs operations and evaluates tensors in fetches.
            This method runs one "step" of TensorFlow computation, by running the necessary graph fragment to execute 
            every Operation and evaluate every Tensor in fetches, substituting the values in feed_dict for the 
            corresponding input values.
        Args:
            fetches: A single graph element, a list of graph elements, or a dictionary whose values are graph elements 
                     or lists of graph elements (described above).
            feed_dict: A dictionary that maps graph elements to values (described above).
            options: A [RunOptions] protocol buffer
            run_metadata: A [RunMetadata] protocol buffer
        Returns:
            Either a single value if fetches is a single graph element, or a list of values if fetches is a list, or a 
            dictionary with the same keys as fetches if that is a dictionary (described above). Order in which fetches 
            operations are evaluated inside the call is undefined.

         ***The fetches argument may be a single graph element, or an arbitrarily nested list, tuple, namedtuple, dict, 
            or OrderedDict containing graph elements at its leaves. A graph element can be one of the following types:
                An tf.Operation. The corresponding fetched value will be None.
                A tf.Tensor. The corresponding fetched value will be a numpy ndarray containing the value of that tensor
                A tf.SparseTensor. The corresponding fetched value will be a tf.SparseTensorValue containing the value 
                    of that sparse tensor.
                A get_tensor_handle op. The corresponding fetched value will be a numpy ndarray containing the handle of 
                    that tensor.
                A string which is the name of a tensor or operation in the graph.
            
            The value returned by run() has the same shape as the fetches argument, where the leaves are replaced by the 
                corresponding values returned by TensorFlow.

            The optional feed_dict argument allows the caller to override the value of tensors in the graph. Each key in
                feed_dict can be one of the following types:
                    If the key is a tf.Tensor, the value may be a Python scalar, string, list, or numpy ndarray that can 
                        be converted to the same dtype as that tensor. Additionally, if the key is a tf.placeholder, the 
                        shape of the value will be checked for compatibility with the placeholder.
                    If the key is a tf.SparseTensor, the value should be a tf.SparseTensorValue.
                    If the key is a nested tuple of Tensors or SparseTensors, the value should be a nested tuple with 
                        the same structure that maps to their corresponding values as above.
            Each value in feed_dict must be convertible to a numpy array of the dtype of the corresponding key.

            The optional options argument expects a [RunOptions] proto. The options allow controlling the behavior of 
                this particular step (e.g. turning tracing on).
            
            The optional run_metadata argument expects a [RunMetadata] proto. When appropriate, the non-Tensor output of 
                this step will be collected there. For example, when users turn on tracing in options, the profiled info 
                will be collected into this argument and passed back.

    """

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

    def learn5(self, iteration):
        episode_observation = len(self.episode_observation)
        self.episode_observation2 = np.array(self.episode_observation).reshape(episode_observation, self.n_features)
        discounted_episode_rewards_norm = self._discount_and_norm_rewards()
        x_batch, y_batch, z_batch = \
            self.next_mini_batch(
                np.vstack(self.episode_observation2),
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
            self.episode_observation, self.episode_actions, self.episode_rewards = [], [], []  # empty episode data

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
