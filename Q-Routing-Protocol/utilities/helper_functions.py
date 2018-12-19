import numpy as np
from envs.simulator import NetworkSimulatorEnv
from agents.q_agent2 import NetworkQAgent, NetworkValAgent


def file_dictionary_extractor(file, printing=True):
    """Extract text and integer-text type from input file as a dictionary"""
    dictionary = {}
    print('test_file =', file)
    with open(file, 'r') as f:
        for line in f.read().splitlines():
            if printing:
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


def q_nn(dictionary, environment, i):
    """Initializes a NetworkQAgent class object to manipulate"""
    feature_set_cardinality = len(environment.resources_bbu + environment.resources_edges)
    x = NetworkQAgent(
        environment.total_nodes,
        i,
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
        dictionary.get('constant_val')[0])
    return x


def val_nn(dictionary, environment, i):
    """Initializes a NetworkValAgent class object to manipulate"""
    feature_set_cardinality = len(environment.resources_bbu + environment.resources_edges)
    x = NetworkValAgent(
        environment.total_nodes,
        i,
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
    )
    return x


def create_agents_lists(arg):
    """Creates a list to hold all relevant information about each agent at each node"""
    list_of_agent_objects = []
    dictionary = arg
    environment = NetworkSimulatorEnv()
    environment.reset_env()
    for nodes in range(0, environment.total_nodes):
        # two agents appended to each node
        list_of_agent_objects.append([q_nn(dictionary, environment, nodes), val_nn(dictionary, environment, nodes)])
    return list_of_agent_objects


def prediction_file(name, data):
    """Writes the numpy data results to a txt file"""
    with open('output_data/%s.txt' % name, 'wb') as outfile:
        for data_slice in data:
            np.savetxt(outfile, data_slice[np.newaxis], fmt='%-7.2f', delimiter=',')
    return outfile

#
# def append_to_agent_lists(iteration, limit, group, speak, array):
#     learning = []
#     if iteration % 1 == 0:
#         for j in range(0, limit):
#             if j not in group:
#                 # agent_list[j].learn_val(iteration)
#                 array[j][1].learn_val(iteration)
#                 array[j][0].learn5(iteration)
#                 if speak:
#                     learning.append(j)
#         if speak:
#             print('learning:', learning, '\n')
#   return array
# append_to_agent_lists(iteration, environment.total_nodes, environment.bbu_connected_nodes, True, agent_objects)

# /t_{}i_{}n_{}sz_{}.m_s_c_d_iat_lr_rb_re_c
# steps
# iters

