import sys
import numpy as np
from datetime import datetime
from envs.simulator import NetworkSimulatorEnv
from agents.q_agent2 import NetworkQAgent, NetworkValAgent


def main(speak=True):

    d = file_dictionary_extractor(sys.argv[1])
    done, data, reward_history = False, [], []

    environment = NetworkSimulatorEnv()
    environment.reset_env()

    # Poisson distributed network model; Requests enter network according to a poisson distribution
    environment.call_mean = d.get('arrival_rate')[0]

    environment.cost = d.get('cost')[0]
    environment.bbu_limit = d.get('resources_bbu')[0]
    environment.edge_limit = d.get('resources_edge')[0]

    list_of_agent_objects = create_agents(d, environment)

    # Have arrival rates be non-stationary

    for iteration in range(d.get('iterations')[0]):
        print("Processing iteration: ", iteration)
        node_destination_tuples = environment.reset_env()
        started = datetime.now()

        for step in range(d.get('time_steps')[0]):
            if not done:
                current_node_destination_pair = node_destination_tuples[1]
                current_node = current_node_destination_pair[0]
                # Action is local edge
                action = list_of_agent_objects[current_node][0].neural_net_action_selection(environment.resources_edges,
                                                                                            environment.resources_bbu)
                node_destination_tuples, done = environment.step(action)

                if step % d.get('dumps')[0] == 0 and step > 0:
                    reward = environment.calculate_reward()
                    reward_history.append(reward)
                    history_queue_length = len(environment.history_queue)
                    current_information = [iteration, step, history_queue_length, environment.send_fail, reward]
                    data.append(list(current_information))

                    if speak:
                        print(current_information)

                    environment.reset_history()

                    # calculate loss
                    for node in range(0, environment.total_nodes):
                        if node not in environment.bbu_connected_nodes:
                            list_of_agent_objects[node][0].store_transition_episode(reward)

        print("Completed in", datetime.now() - started)

        learning = []
        if iteration % 1 == 0:
            for j in range(0, environment.total_nodes):
                if j not in environment.bbu_connected_nodes:
                    # agent_list[j].learn_val(iteration)
                    list_of_agent_objects[j][0].learn5(iteration)
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
    print('test_file =', test_file)
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
    return dictionary


def create_agents(dictionary, environment):
    list_of_agent_objects = []
    feature_set_cardinality = len(environment.resources_bbu + environment.resources_edges)
    for nodes in range(0, environment.total_nodes):
        """
        Create a list to hold lots of relevant information for each agent at their respective nodes. 
        The relevant information has those method names given in the q_agent.py script.
        There are 37 objects in these lists as of 11/20/2018.
        """
        # two agents appended to each node
        list_of_agent_objects.append(
            [NetworkQAgent(
                environment.total_nodes,
                nodes,
                environment.total_edges_from_node,
                environment.node_to_node,
                environment.absolute_node_edge_tuples,
                environment.bbu_connected_nodes,
                feature_set_cardinality,
                dictionary.get('learning_rate')[0],
                dictionary.get('number_layers')[0],
                dictionary.get('layer_types'),
                dictionary.get('mean_value')[0],
                dictionary.get('std_val')[0],
                dictionary.get('constant_val')[0]),
                NetworkValAgent(
                    environment.total_nodes,
                    nodes,
                    environment.total_edges_from_node,
                    environment.node_to_node,
                    environment.absolute_node_edge_tuples,
                    environment.bbu_connected_nodes,
                    feature_set_cardinality,
                    dictionary.get('learning_rate')[0],
                    dictionary.get('number_layers')[0],
                    dictionary.get('layer_types'),
                    dictionary.get('mean_value')[0],
                    dictionary.get('std_val')[0],
                    dictionary.get('constant_val')[0]
                )]
        )
    return list_of_agent_objects

# def plot_display(file):
#     filename = file
#     data_df = pd.read_csv(filename)
#     reward_data = data_df['calculated_reward'].values
#     plt.plot(reward_data)
#     plt.show()


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

if __name__ == '__main__':
    main()
