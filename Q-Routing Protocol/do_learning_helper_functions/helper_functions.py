import numpy as np
from envs.simulator import NetworkSimulatorEnv
from agents.q_agent2 import NetworkQAgent, NetworkValAgent


def file_dictionary_extractor(file, printing=True):
    test_file, dictionary = file, {}
    print('test_file =', test_file)
    with open(test_file, 'r') as f:
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


def create_agents():
    """
    Create a list to hold lots of relevant information for each agent at their respective nodes.
    The relevant information has those method names given in the q_agent.py script.
    There are 37 objects in these lists as of 11/20/2018.
    """
    list_of_agent_objects = []
    dictionary = file_dictionary_extractor('input_data/TestPar1.txt')
    environment = NetworkSimulatorEnv()
    environment.reset_env()
    for nodes in range(0, environment.total_nodes):
        # two agents appended to each node
        list_of_agent_objects.append([q_nn(dictionary, environment, nodes), val_nn(dictionary, environment, nodes)])
    return list_of_agent_objects


def prediction_file(data):
    with open('output_data/predictions.txt', 'wb') as outfile:
        for data_slice in data:
            np.savetxt(outfile, data_slice[np.newaxis], fmt='%-7.2f', delimiter=',')
    return outfile
