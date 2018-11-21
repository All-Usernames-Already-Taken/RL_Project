import numpy as np
from envs.simulator import NetworkSimulatorEnv
from agents.q_agent import NetworkTabularQAgent
from sys import argv


def main(speak=True):
    # speak = False

    d, test_file = file_dictionary_extractor(argv[1])

    time_steps = d['time_steps'][0]
    iterations = d['iterations'][0]
    total_layers = d['number_layers'][0]
    layer_types = d['layer_types']
    mean_val = d['mean_value']
    std_val = d['std_val']
    constant_val = d['constant_val']
    layer_sizes = d['layer_sizes']
    dumps = d['dumps'][0]
    arrival_rate = d['arrival_rate'][0]
    learning_rate = d['learning_rate'][0]
    resources_bbu = d['resources_bbu'][0]
    resources_edge = d['resources_edge'][0]  # Num of fiber channels
    cost = d['cost'][0]
    act_func = d['act']

    data, r_hist, agent_list = ([],) * 3

    # Poisson distributed network model
    call_mean = arrival_rate

    # call_mean += 0
    env = NetworkSimulatorEnv()
    env.reset()

    env.call_mean = call_mean
    env.bbu_limit = resources_bbu
    env.edge_limit = resources_edge
    env.cost = cost

    # cg: set up agents for every node
    # The following code line gives 13 for 30 and 10, bbu and edge resources respectively.
    # Adding arrays means we get a larger component-wise array
    n_features = len(env.resources_bbu + env.resources_edges)

    done = False
    for nodes in range(0, env.total_nodes):
        """
        Create a list to hold lots of relevant information for each agent at their respective nodes. 
        The relevant information has those method names given in the q_agent.py script.
        There are 37 objects in these lists as of 11/20/2018.
        """
        agent_list.append(
            NetworkTabularQAgent(
                env.total_nodes,
                env.total_edges,
                nodes,
                env.total_edges_from_node,
                env.node_to_node,
                env.absolute_node_absolute_edge_tuples,
                env.bbu_connected_nodes,
                n_features,
                learning_rate,
                total_layers,
                layer_sizes,
                layer_types,
                mean_val,
                std_val,
                constant_val,
                act_func
            )
        )

        # Have arrival rates be nonstationary

    for iteration in range(iterations):
        state_pair = env.reset()
        for t in range(time_steps):
            if not done:
                current_state = state_pair[1]
                n = current_state[0]
                action = agent_list[n].act_nn2(env.resources_edges, env.resources_bbu)  # Action is local edge
                state_pair, done = env.step(action)
                if t % dumps == 0 and t > 0:
                    if speak:
                        print(
                            "iteration: {}\n"
                            "time: {}\n"
                            "send_fail: {}\n"
                            "history queue length: {}\n"
                            "calculated_reward: {}\n\n".format(
                                iteration,
                                t,
                                env.send_fail,
                                len(env.history_queue),
                                env.calculate_reward()
                            )
                        )
                    reward = env.calculate_reward()
                    r_hist.append(reward)
                    data.append(
                        [iteration, t, len(env.history_queue), env.send_fail, reward])
                    env.reset_history()

                    # calculate loss
                    for j in range(0, env.total_nodes):
                        if j not in env.bbu_connected_nodes:
                            agent_list[j].store_transition_episode(reward)

        if iteration % 1 == 0:
            if speak:
                print("learning")
            for j in range(0, env.total_nodes):
                if j not in env.bbu_connected_nodes:
                    if speak:
                        print(j)
                    agent_list[j].learn5(iteration)

        # Record statistics from iteration
        # (routed_packets, send fails, average number of hops, average completion time, max completion time)
        # Learn/backpropagation

    predictive_file = 'predictions' + test_file.split('.txt')[0]
    data = np.array(data)
    with open(predictive_file, 'wb') as outfile:

        # outfile.write('# Array shape: {0}\n'.format(data.shape))
        # Iterating through n-D array produces slices along the last axis.
        # Here, data[i,:,:] is equivalent

        for data_slice in data:
            np.savetxt(outfile, data_slice[np.newaxis], fmt='%-7.2f', delimiter=',')

            # Writing out a break to indicate different slices...
            # outfile.write('# New slice\n')


def file_dictionary_extractor(file):
    test_file, d = file, {}
    with open(test_file, 'r') as f:
        for line in f.read().splitlines():
            print(line)
            line = line.strip()
            k, v = line.split(':')
            k, v = k.strip(), v.strip()
            v = v.split(',')
            v = [v[i].strip() for i in range(len(v))]
            for j in range(len(v)):
                try:
                    v[j] = int(v[j])
                except ValueError:
                    try:
                        v[j] = float(v[j])
                    except ValueError:
                        pass
                    pass
            d.setdefault(k, v)
    return d, test_file


if __name__ == '__main__':
    main()
