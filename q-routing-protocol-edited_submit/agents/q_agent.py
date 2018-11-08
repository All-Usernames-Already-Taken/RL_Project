import collections
import numpy as np
from random import random
import tensorflow as tf

class networkTabularQAgent(object):
 """
 Agent implementing tabular Q-learning for the NetworkSimulatorEnv.
 """

 def __init__(self, num_nodes, num_actions, node, nlinks, links, link_num, dests, n_features, learning_rate, num_layers, layer_size, layer_type, mean_val, std_val, constant_val, activation_type):

  #cg: reset configuration for each node in the graph
  self.config = {
   "init_mean": 0.0,  # Initialize Q values with this mean
   "init_std": 0.0,   # Initialize Q values with this standard deviation
   "learning_rate" : 0.7, 
   "eps": 0.1,      # Epsilon in epsilon greedy policies
   "discount": 1, 
   "n_iter": 10000000}  # Number of iterations
  #self.q = np.zeros((num_nodes, num_nodes, num_actions))

  self.hist_resources = []
  self.hist_action = []
  self.n = node
  self.links = links
  self.link_num = link_num
  self.dests = dests
  #self.nlinks = nlinks
  self.n_actions = nlinks[self.n]
  self.n_features = n_features
  self.ep_obs, self.ep_as, self.ep_rs = [], [], []
  self.ep_obs_temp, self.ep_as_temp = [], []
  self.lr = learning_rate  #learning_rate


  self.sess = tf.Session()

  observations = tf.placeholder(shape = [None, self.n_actions], dtype = tf.float32)
  actions = tf.placeholder(shape = [None], dtype = tf.float32)
  rewards = tf.placeholder(shape = [None], dtype = tf.float32)

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
  	self.neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis = 1)
   self.loss = tf.reduce_mean(self.neg_log_prob * self.tf_vt) # reward guided loss
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
    )
   if layer_type[i] == 'last':
    self.all_act = tf.layers.dense(
     inputs = temp_layer_in, 
     units = self.n_actions, 
     activation = act[activation_type[i]], 
     kernel_initializer = tf.random_normal_initializer(mean = mean_val[i], stddev = std_val[i]), 
     bias_initializer = tf.constant_initializer(constant_val[i])
    )
   temp_layer_in = temp_layer

  self.all_act_prob = tf.nn.softmax(self.all_act) # use softmax to convert to probability

  with tf.name_scope('loss'):
   self.neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis = 1)
   self.loss = tf.reduce_mean(self.neg_log_prob * self.tf_vt) # reward guided loss
   print("help")
  with tf.name_scope('train'):
   self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

 def choose_action(self, observation, valid):
  prob_weights = self.sess.run(self.all_act_prob, feed_dict = {self.tf_obs:observation})
  valid_weights = prob_weights*valid
  valid_prob = self.normalize_weights(valid_weights[0])
  action = np.random.choice(range(prob_weights.shape[1]), p = valid_prob.ravel())
  return action

 def choose_action2(self, observation):
  prob_weights = self.sess.run(self.all_act_prob, feed_dict = {self.tf_obs:observation})
  action = np.random.choice(range(prob_weights.shape[1]), p = prob_weights.ravel())
  return action

 def store_transition(self, s, a, r):
  self.ep_obs.append(s)
  self.ep_as.append(a)
  self.ep_rs.append(r)

 def store_transition_temp(self, s, a):
  self.ep_obs_temp.append(s)
  self.ep_as_temp.append(a)

 def store_transition_episode(self, r):
  l = len(self.ep_as_temp)
  for i in range(0, l):
   self.store_transition(self.ep_obs_temp[i], self.ep_as_temp[i], r)


 def learn2(self):
  l = len(self.ep_obs)
  self.ep_obs = np.array(self.ep_obs).reshape(l, self.n_features)
  self.sess.run(
   self.train_op, 
   feed_dict = {
    self.tf_obs: np.vstack(self.ep_obs), # shape = [None, n_obs]
    self.tf_acts: np.array(self.ep_as),  # shape = [None, ]
    self.tf_vt: np.array(self.ep_rs),   # shape = [None, ]
  })

  self.ep_obs, self.ep_as, self.ep_rs = [], [], [] # empty episode data

 def learn3(self, iter):
  l = len(self.ep_obs)
  self.ep_obs2 = np.array(self.ep_obs).reshape(l, self.n_features)
  discounted_ep_rs_norm = self._discount_and_norm_rewards()
  _, loss, log_probs, act_val, l1 = self.sess.run(
   [self.train_op, self.loss, self.neg_log_prob, self.all_act, self.layer], 
   feed_dict = {
    self.tf_obs: np.vstack(self.ep_obs2),     # shape = [None, n_obs]
    self.tf_acts: np.array(self.ep_as),      # shape = [None, ]
    self.tf_vt: np.array(discounted_ep_rs_norm), # shape = [None, ]
    })
  if iter%2 == 0:
   self.ep_obs, self.ep_as, self.ep_rs = [], [], [] # empty episode data

 def learn5(self, iter):
  l = len(self.ep_obs)
  self.ep_obs2 = np.array(self.ep_obs).reshape(l, self.n_features)
  discounted_ep_rs_norm = self._discount_and_norm_rewards()
  X_batch, Y_batch, Z_batch = self.next_minibatch(
   np.vstack(self.ep_obs2), 
   np.array(self.ep_as), 
   np.array(discounted_ep_rs_norm), 
   l)
  _, loss, log_probs, act_val = self.sess.run(
   [self.train_op, self.loss, self.neg_log_prob, self.all_act], 
   feed_dict = {
    self.tf_obs: X_batch,  # shape = [None, n_obs]
    self.tf_acts: Y_batch, # shape = [None, ]
    self.tf_vt: Z_batch,  # shape = [None, ]
   })
  if iter%1 == 0:
   self.ep_obs, self.ep_as, self.ep_rs = [], [], [] # empty episode data

 def learn4(self, iter):
  l = len(self.ep_obs)
  self.ep_obs2 = np.array(self.ep_obs).reshape(l, self.n_features)
  for i in range(0, 1):
   X_batch, Y_batch, Z_batch = self.next_minibatch(
    np.vstack(self.ep_obs2), 
    np.array(self.ep_as), 
    np.array(self.ep_rs), 
    l)
   self.sess.run(
    self.train_op, 
    feed_dict = {
     self.tf_obs: X_batch,  # shape = [None, n_obs]
     self.tf_acts: Y_batch, # shape = [None, ]
     self.tf_vt: Z_batch,  # shape = [None, ]
   })
  if iter%5 == 0:
   self.ep_obs, self.ep_as, self.ep_rs = [], [], [] # empty episode data

 def _discount_and_norm_rewards(self):
  self.gamma = 0
  discounted_ep_rs = np.zeros_like(self.ep_rs)
  running_add = 0
  for t in reversed(range(0, len(self.ep_rs))):
   running_add = running_add * self.gamma + self.ep_rs[t]
   discounted_ep_rs[t] = running_add

  discounted_ep_rs -= np.mean(discounted_ep_rs)
  discounted_ep_rs /= np.std(discounted_ep_rs)
  return discounted_ep_rs

 def act_nn(self, resources_edges, resources_bbu):
  obs = resources_edges+resources_bbu
  obs = np.array(obs).reshape(1, self.n_features)
  valid = np.zeros(self.n_actions)
  flag_e = 1
  flag_b = 1
  for i in range(0, self.n_actions):
   if resources_edges[self.link_num[self.n][i]] > 0:
    flag_e = 0
    valid[i] = 1
  for i in resources_bbu:
   if i > 0:
    flag_b = 0
  #if resources_bbu > 0:
   # flag_b = 0
  if flag_b or flag_e:
   action = -1
  else:
   flag = 1
   while flag:
    #action = int(np.random.choice(range(0, nlinks[n]))) # choose random action
    action = self.choose_action(obs, valid)
    #self.store_transition_temp(resources_edges + resources_bbu, action)
    next_node = self.links[self.n][action]
    l_num = self.link_num[self.n][action]
    if next_node in self.dests:
     if resources_bbu[self.dests.index(next_node)] == 0:
      flag = 1
      valid[action] = 0
     else:
      flag = 0
    else:
     flag = 0

    if resources_edges[self.link_num[self.n][action]] == 0:
       flag = 1

  #self.hist_action.append((resources_edges+resources_bbu, action))
  if action >= 0:
    self.store_transition_temp(resources_edges+resources_bbu, action)
  return action

 def act_nn2(self, resources_edges, resources_bbu):
  obs = resources_edges+resources_bbu
  obs = np.array(obs).reshape(1, self.n_features)
  action = self.choose_action2(obs)
  self.store_transition_temp(resources_edges+resources_bbu, action)
  next_node = self.links[self.n][action]
  l_num = self.link_num[self.n][action]
  if resources_edges[self.link_num[self.n][action]] == 0:
   action = -1
  elif next_node in self.dests:
   if resources_bbu[self.dests.index(next_node)] == 0:
    action = -1
  return action


 def act(self, state, nlinks, links, resources_edges, resources_bbu, link_num, dests, best = False):
  n = state[0]
  flag_e = 1
  flag_b = 1
  for i in range(0, nlinks[n]):
   if resources_edges[link_num[n][i]] > 0:
    flag_e = 0
  for i in resources_bbu:
   if i > 0:
    flag_b = 0
  #if resources_bbu > 0:
   # flag_b = 0
  if flag_b or flag_e:
   action = -1
  else:
   flag = 1
   while flag:
    action = int(np.random.choice(range(0, nlinks[n]))) #choose random action
    next_node = links[n][action]
    l_num = link_num[n][action]
    if next_node in dests:
     if resources_bbu[dests.index(next_node)] == 0:
      flag = 1
     else:
      flag = 0
    else:
     flag = 0
    if resources_edges[link_num[n][action]] == 0:
       flag = 1

  self.hist_action.append((resources_edges+resources_bbu, action))

  return action


 def learn(self, current_event, next_event, reward, action, done, nlinks):
  # NN
  n = current_event[0]
  dest = current_event[1]

  n_next = next_event[0]
  dest_next = next_event[1]

  future = self.q[n_next][dest][0]
  for link in range(nlinks[n_next]):
   if self.q[n_next][dest][link] < future:
    future = self.q[n_next][dest][link]

  # Q learning
  self.q[n][dest][action] += (reward + self.config["discount"]*future - self.q[n][dest][action])* self.config["learning_rate"]



