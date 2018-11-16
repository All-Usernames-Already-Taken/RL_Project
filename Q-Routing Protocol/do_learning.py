import numpy as np
from envs.simulator import NetworkSimulatorEnv
from agents.q_agent import NetworkTabularQAgent
from sys import argv


def main():

    test_file = argv[1]
    d = {}
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

    time_steps = d['time_steps'][0]
    iterations = d['iterations'][0]
    num_layers = d['number_layers'][0]
    layer_types = d['layer_types']
    mean_val = d['mean_value']
    std_val = d['std_val']
    constant_val = d['constant_val']
    layer_sizes = d['layer_sizes']
    dumps = d['dumps'][0]
    arrival_rate = d['arrival_rate'][0]
    learning_rate = d['learning_rate'][0]
    resources_bbu = d['resources_bbu'][0]
    resources_edge = d['resources_edge'][0]
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
    n_features = len(env.resources_bbu + env.resources_edges)
    for i in range(0, env.total_nodes):

        # agent_list.append(
        #     NetworkTabularQAgent(
        #         env.nnodes,
        #         env.nedges,
        #         env.distance,
        #         env.nlinks))

        agent_list.append(
            NetworkTabularQAgent(
                env.total_nodes,
                env.total_edges,
                i,
                env.total_local_connections,
                env.links,
                env.abs_link_id,
                env.destinations,
                n_features,
                learning_rate,
                num_layers,
                layer_sizes,
                layer_types,
                mean_val,
                std_val,
                constant_val,
                act_func))

    done = False
    for i in range(iterations):

        # call_mean += 0
        # env = NetworkSimulatorEnv()
        # r_sum_random = r_sum_best = 0

        state_pair = env.reset()

        for t in range(time_steps):
            if not done:
                current_state = state_pair[1]
                n = current_state[0]

                action = agent_list[n].act_nn2(env.resources_edges, env.resources_bbu)
                state_pair, done = env.step(action)

                if t % dumps == 0 and t > 0:
                    print("iteration: {}\n"
                          "time: {}\n"
                          "send_fail: {}\n"
                          "history queue length: {}\n"
                          "calculated_reward: {}\n\n"
                          .format(i, t, env.send_fail, len(env.history_queue), env.calculate_reward()))
                    r = env.calculate_reward()
                    r_hist.append(r)
                    data.append(
                        [i, t, len(env.history_queue), env.send_fail, r])
                    env.reset_history()

                    # calculate loss
                    for j in range(0, env.total_nodes):
                        if j not in env.destinations:
                            agent_list[j].store_transition_episode(r)

        if i % 1 == 0:
            print("learning")
            for j in range(0, env.total_nodes):
                if j not in env.destinations:
                    print(j)
                    agent_list[j].learn5(i)

        # Record statistics from iteration
        # (routed_packets, send fails, average number of hops, average completion time, max completion time)
        # Learn/backpropagation

    pred_file = 'predictions' + test_file.split('.txt')[0]
    data = np.array(data)
    with open(pred_file, 'wb') as outfile:

        # outfile.write('# Array shape: {0}\n'.format(data.shape))
        # Iterating through n-D array produces slices along the last axis.
        # Here, data[i,:,:] is equivalent

        for data_slice in data:
            np.savetxt(outfile, data_slice[np.newaxis], fmt='%-7.2f', delimiter=',')

            # Writing out a break to indicate different slices...
            # outfile.write('# New slice\n')


if __name__ == '__main__':
    main()

# HELLO THERE!!!!
