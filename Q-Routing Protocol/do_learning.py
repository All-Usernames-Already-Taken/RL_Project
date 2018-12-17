from agents.q_agent2 import NetworkQAgent, NetworkValAgent
from envs.simulator import NetworkSimulatorEnv
from datetime import datetime
import os

# from output_data.data_manipulation import display_data
import pandas as pd
import matplotlib.pyplot as plt
import csv
import sys
import numpy as np


def main(speak=True):

    done = False

    d, test_file = file_dictionary_extractor(sys.argv[1])

    time_steps = d.get('time_steps')[0]
    episodes = d.get('iterations')[0]
    total_layers = d.get('number_layers')[0]
    layer_types = d.get('layer_types')
    mean_val = d.get('mean_value')[0]
    std_val = d.get('std_val')[0]
    constant_val = d.get('constant_val')[0]
    layer_sizes = d.get('layer_sizes')
    dumps = d.get('dumps')[0]
    arrival_rate = d.get('arrival_rate')[0]
    learning_rate = d.get('learning_rate')[0]
    resources_bbu = d.get('resources_bbu')[0]
    resources_edge = d.get('resources_edge')[0]  # Number of channels per fiber
    cost = d.get('cost')[0]

    data = []
    reward_history = []
    agent_list = []

    environment = NetworkSimulatorEnv()
    # environment.reset()
    environment.reset_env()
    environment.cost = cost

    # Poisson distributed network model
    environment.call_mean = arrival_rate
    environment.bbu_limit = resources_bbu
    environment.edge_limit = resources_edge

    # cg: set up agents for every node
    # The following code line gives 13 for 30 and 10, bbu and edge resources respectively.
    # Adding arrays means we get a larger component-wise array, not vector addition
    feature_set_cardinality = len(environment.resources_bbu + environment.resources_edges)

    for nodes in range(0, environment.total_nodes):
        """
        Create a list to hold lots of relevant information for each agent at their respective nodes. 
        The relevant information has those method names given in the q_agent.py script.
        There are 37 objects in these lists as of 11/20/2018.
        """
        #two agents appended to each node
        val_nn = NetworkValAgent(
                    environment.total_nodes,
                    nodes,
                    environment.total_edges_from_node,
                    environment.node_to_node,
                    environment.absolute_node_edge_tuples,
                    environment.bbu_connected_nodes,
                    feature_set_cardinality,
                    learning_rate,
                    total_layers,
                    layer_types,
                    mean_val,
                    std_val,
                    constant_val,
                )

        agent_list.append(
            [NetworkQAgent(
                environment.total_nodes,
                nodes,
                environment.total_edges_from_node,
                environment.node_to_node,
                environment.absolute_node_edge_tuples,
                environment.bbu_connected_nodes,
                feature_set_cardinality,
                learning_rate,
                total_layers,
                layer_types,
                mean_val,
                std_val,
                constant_val,
                val_nn
            ),
                val_nn
                ]
        )

    # Have arrival rates be non-stationary

    # with open('input_data/results-%s.csv' % start_time.strftime("%Y-%m-%d %H:%M"), 'w+') as csv_file:
    #     data_writer = csv.writer(csv_file, delimiter=',')
    #     data_writer.writerow(['episodes', 'time_step', 'history_queue_length', 'send_fail', 'calculated_reward'])

    for iteration in range(episodes):
        print("Processing iteration: ", iteration)
        state_pair = environment.reset_env()
        started = datetime.now()
        for t in range(time_steps):
            if not done:
                current_node = state_pair[1]
                n = current_node[0]
                action = agent_list[n][0].act_nn2(environment.resources_edges,environment.resources_bbu)  # Action is local edge
                state_pair, done = environment.step(action)
                # val =
                if t % dumps == 0 and t > 0:
                    reward = environment.calculate_reward()
                    reward_history.append(reward)
                    history_queue_length = len(environment.history_queue)
                    current_information = [iteration, t, history_queue_length, environment.send_fail, reward]

                    data.append(list(current_information))

                    if speak:
                        print(current_information)

                    # data_writer.writerow(current_information)

                    environment.reset_history()

                    # calculate loss
                    for node in range(0, environment.total_nodes):
                        if node not in environment.bbu_connected_nodes:
                            agent_list[node][0].store_transition_episode(reward)
                            agent_list[node][1].store_transition_episode(reward)

        print("Completed in", datetime.now() - started)

        learning = []
        if iteration % 1 == 0:
            for j in range(0, environment.total_nodes):
                if j not in environment.bbu_connected_nodes:
                    agent_list[j][1].learn_val(iteration)
                    agent_list[j][0].learn5(iteration)
                    if speak:
                        learning.append(j)
            if speak:
                print('learning:', learning, '\n')

            # Record statistics from iteration
            # (routed_packets, send fails, average number of hops, average completion time, max completion time)
            # Learn/backpropagation

    data = np.array(data)
    with open('output_data/predictions.txt', 'wb') as outfile:
        # outfile.write('# Array shape: {0}\n'.format(input_data.shape))
        # Iterating through n-D array produces slices along the last axis.
        # Here, input_data[i,:,:] is equivalent
        for data_slice in data:
            np.savetxt(outfile, data_slice[np.newaxis], fmt='%-7.2f', delimiter=',')

    # plot_display('output_data/predictions.txt')

    # Writing out a break to indicate different slices...
    # outfile.write('# New slice\n')


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def file_dictionary_extractor(file):
    test_file, dictionary = file, {}
    print('test_file =',test_file)
    with open(test_file, 'r') as f:
        for line in f.read().splitlines():
            print(line)
            line = line.strip()
            key, value = line.split(':')
            key, value = key.strip(), value.strip()
            value = value.split(',')
            value = [value[i].strip() for i in range(len(value))]
            for j in range(len(value)):
                try:
                    value[j] = int(value[j])
                except ValueError:
                    try:
                        value[j] = float(value[j])
                    except ValueError:
                        value[j] = str(value[j])
                        pass
            dictionary.setdefault(key, value)
    return dictionary, test_file


# def plot_display(file):
#     filename = file
#     data_df = pd.read_csv(filename)
#     reward_data = data_df['calculated_reward'].values
#     plt.plot(reward_data)
#     plt.show()


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


if __name__ == '__main__':
    main()
