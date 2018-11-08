import gym
import numpy as np
from gym import spaces, envs
from envs.simulator import NetworkSimulatorEnv
from agents.q_agent import networkTabularQAgent
import sys

def main():
	##

	test_file=sys.argv[1]
	inputfile=open(test_file)
	my_text = inputfile.readlines()
	timesteps=int(my_text[1].replace("\n","").split(":")[1])
	iterations=int(my_text[2].replace("\n","").split(":")[1])
	num_layers=int(my_text[3].replace("\n","").split(":")[1])
	layer_types0=my_text[4].replace("\n","").split(":")[1]
	layer_types=layer_types0.replace(" ","").split(",")
	layer_sizes0=my_text[5].replace("\n","").split(":")[1]
	layer_sizes=layer_sizes0.split(",")
	for i in range(len(layer_sizes)):
		layer_sizes[i]=int(layer_sizes[i])
	mean_val0=my_text[6].replace("\n","").split(":")[1]
	mean_val=mean_val0.split(",")
	for i in range(len(mean_val)):
		mean_val[i]=float(mean_val[i])
	std_val0=my_text[7].replace("\n","").split(":")[1]
	std_val=std_val0.split(",")
	for i in range(len(std_val)):
		std_val[i]=float(std_val[i])
	constant_val0=my_text[8].replace("\n","").split(":")[1]
	constant_val=constant_val0.split(",")
	for i in range(len(constant_val)):
		constant_val[i]=float(constant_val[i])
	dumps=int(my_text[9].replace("\n","").split(":")[1])
	arrival_rate=int(my_text[10].replace("\n","").split(":")[1])
	learning_rate=float(my_text[11].replace("\n","").split(":")[1])
	resources_bbu=int(my_text[12].replace("\n","").split(":")[1])
	resources_edge=int(my_text[13].replace("\n","").split(":")[1])
	cost=int(my_text[14].replace("\n","").split(":")[1])
	act_func0=my_text[15].replace("\n","").split(":")[1]
	act_func=act_func0.split(",")
	for i in range(len(act_func)):
		act_func[i]=int(act_func[i])


	# In[180]:


	print("Timesteps: %s " % str(timesteps)) #check
	print("Iterations: %s " % str(iterations)) #check
	print("Number Layers: %s " % str(num_layers))
	print("layer_types: %s " % str(layer_types))
	print("mean_val: %s " % str(mean_val))
	print("std_val: %s " % str(std_val))
	print("constant_val: %s " % str(constant_val))
	print("layer_sizes: %s " % str(layer_sizes))
	print("dumps: %s" % str(dumps)) #check
	print("arrival_rate: %s" % str(arrival_rate)) #check
	print("learning_rate: %s" % str(learning_rate))
	print("resources_bbu: %s" % str(resources_bbu)) #check
	print("resources_edge: %s" % str(resources_edge)) #check
	print("cost: %s" % str(cost))
	print("act: %s" % str(act_func))#check
	#set seeds and finish imports


	##
	data=[]
	r_hist=[]
	callmean = arrival_rate
	#callmean += 0
	env = NetworkSimulatorEnv()
	state_pair = env.reset()
	env.callmean = callmean
	env.bbu_limit=resources_bbu
	env.edge_limit=resources_edge
	env.cost=cost
	#cg: set up agents for every node
	agent_list=[]
	n_features=len(env.resources_bbu+env.resources_edges)
	for i in range(0,env.nnodes):
		#agent_list.append(networkTabularQAgent(env.nnodes, env.nedges, env.distance, env.nlinks))
		agent_list.append(networkTabularQAgent(env.nnodes, env.nedges, i, env.nlinks,env.links,env.link_num,env.dests,n_features,learning_rate, num_layers,layer_sizes,layer_types,mean_val,std_val,constant_val,act_func))
		#num_nodes, num_actions,  node, nlinks, links,  link_num, dests,n_features)
	config = agent_list[i].config
	#agent = networkTabularQAgent(env.nnodes, env.nedges, env.distance, env.nlinks)
	done = False
	for i in range(iterations):
		#callmean += 0
		#env = NetworkSimulatorEnv()
		state_pair = env.reset()
		r_sum_random = r_sum_best = 0


		for t in range(timesteps):
			if not done:

				current_state = state_pair[1]
				n = current_state[0]
				dest = current_state[1]
				#cg: dont need next 3 lines, go directly to act
				#for action in xrange(env.nlinks[n]):
				#    reward, next_state = env.pseudostep(action)
				#    agent.learn(current_state, next_state, reward, action, done, env.nlinks)

				#action  = agent_list[n].act(current_state, env.nlinks,env.links, env.resources_edges,env.resources_bbu,env.link_num,env.dests)
				action=agent_list[n].act_nn2(env.resources_edges,env.resources_bbu)
				state_pair, done = env.step(action)

				next_state = state_pair[0]

				if t%dumps==0 and t>0:
					print("iteration")
					print(i)
					print("time")
					print(t)
					print(env.send_fail)
					print(len(env.history_queue))
					print(env.calculate_reward())
					r=env.calculate_reward()
					r_hist.append(r)
					data.append([i,t,len(env.history_queue),env.send_fail,r])
					env.reset_history()
					#calculate loss
					for j in range(0,env.nnodes):
						if j not in env.dests:
							agent_list[j].store_transition_episode(r)


		if i%1==0:
			print("learning")
			for j in range(0,env.nnodes):
				if j not in env.dests:
					print(j)
					agent_list[j].learn5(i)
		#record statistics from iteration (routed_packets, send fails, average number of hops, average completion time, max completion time)
		#learn/backpropagation
	pred_file='predictions'+test_file.split('.txt')[0]
	data=np.array(data)
	with open(pred_file, 'wb') as outfile:
		#outfile.write('# Array shape: {0}\n'.format(data.shape))
		# Iterating through a ndimensional array produces slices along
		# the last axis. This is equivalent to data[i,:,:] in this case
		for data_slice in data:

			# The formatting string indicates that I'm writing out
			# the values in left-justified columns 7 characters in width
			# with 2 decimal places.
			np.savetxt(outfile, data_slice[np.newaxis], fmt='%-7.2f', delimiter=',')

			# Writing out a break to indicate different slices...
			#outfile.write('# New slice\n')
if __name__ == '__main__':
	main()
