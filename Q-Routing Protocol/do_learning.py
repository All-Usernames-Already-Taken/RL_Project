import numpy as np
from datetime import datetime
from envs.simulator import NetworkSimulatorEnv
from do_learning_helper_functions.helper_functions import file_dictionary_extractor as fde
from do_learning_helper_functions.helper_functions import create_agents_lists as cal
from do_learning_helper_functions.helper_functions import prediction_file as pf


def main(speak=True):
    speak=False
    # The input parameter in the configuration path is now obsolete
    d = fde('input_data/TestPar1.txt')
    done, data, reward_history = False, [], []

    environment = NetworkSimulatorEnv()
    environment.reset_env()

    # --> Removing the next 6 lines for some reason results in all zero or Nan output
    # Requests enter network according to a poisson distribution
    environment.call_mean = d.get('arrival_rate')[0]
    environment.cost = d.get('cost')[0]
    environment.bbu_limit = d.get('resources_bbu')[0]
    environment.edge_limit = d.get('resources_edge')[0]

    """Create list of 9 nodes each with two neural networks"""
    agent_list = cal()

    for iteration in range(d.get('iterations')[0]):
        print("PROCESSING ITERATION: {}\n".format(iteration))

        """Reset environment for each epoch, and record startTime of epoch"""
        node_destination_tuples = environment.reset_env()
        started = datetime.now()

        for step in range(d.get('time_steps')[0]):
            # UNLESS TERMINAL STATE REACHED
            if not done:
                current_node_destination_pair = node_destination_tuples[1]
                current_node = current_node_destination_pair[0]
                # Action is local edge
                """Store the action in the Q Network agent"""
                action = agent_list[current_node][0].act_nn2(environment.resources_edges, environment.resources_bbu)

                """Store the environment state and action"""
                agent_list[current_node][1].store_transition_temp(environment.resources_edges+environment.resources_bbu)

                """Execute action and store information tracking the request for next iteration"""
                node_destination_tuples, done = environment.step(action)

                # EVERY 50 STEPS CALCULATE CUMULATIVE REWARD AND LOSS
                if step % d.get('dumps')[0] == 0 and step > 0:
                    """Calculate reward every dumps size, and track reward history"""
                    reward = environment.calculate_reward()
                    reward_history.append(reward)
                    history_queue_length = len(environment.history_queue)
                    current_information = [iteration, step, history_queue_length, environment.send_fail, reward]
                    data.append(list(current_information))

                    if speak:
                        print(current_information)

                    environment.reset_history()

                    """Calculate loss for each node"""
                    for node in range(0, environment.total_nodes):
                        if node not in environment.bbu_connected_nodes:
                            agent_list[node][0].store_transition_episode(reward)
                            agent_list[node][1].store_transition_episode(reward)

        print("Completed in", datetime.now() - started)

        # Record statistics from iteration
        # (routed_packets, send fails, average number of hops, average completion time, max completion time)
        # Learn/backpropagation
        learning = []
        if iteration % 1 == 0:
            for j in range(0, environment.total_nodes):
                if j not in environment.bbu_connected_nodes:
                    # agent_list[j].learn_val(iteration)
                    agent_list[j][1].learn_val(iteration)
                    val_approx = agent_list[j][1].eval_nn(environment.resources_edges, environment.resources_bbu)
                    print("node:{}, val_approx: {}".format(j, val_approx))
                    agent_list[j][0].learn5(iteration, val_approx)
                    if speak:
                        learning.append(j)
            if speak:
                print('learning:', learning, '\n')

    data = np.array(data)
    pf(data)


if __name__ == '__main__':
    main()
