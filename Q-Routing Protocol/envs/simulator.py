# based on code: https://github.com/Duncanswilson/q-routing-protocol
from abc import ABC
from collections import defaultdict
from heapq import heappush, heappop
from math import log as m_log
from random import random
from numpy import zeros
import gym


events = 0


# Event is fiber path usage
class Event:
    def __init__(self, time, src):
        self.birth = time
        self.destination = UNKNOWN
        self.event_time = time
        self.hops = 0
        self.lifetime = 10
        self.node = src
        self.q_time = 0
        self.resources = []
        self.source = UNKNOWN
        # ?! --> need to add status for path taken, and which bbu resource used
        # ?! --> what are self.source and self.src that they are different?


# Special events
"""INJECT signifies adding a new packet"""
INJECT, REPORT, END_SIM, UNKNOWN = -1, -2, -3, -4
NIL = Nil = -1


class NetworkSimulatorEnv(gym.Env, ABC):
    def __init__(self):
        self.absolute_node_edge_tuples = defaultdict(dict)
        self.active_packets = 0
        self.bbu_connected_nodes = [3, 5]
        self.bbu_limit = 0
        self.call_mean = 5  # Network load
        self.cost = 0
        self.current_event = Event(0.0, 0)  # do I need to do this?
        self.distance = []
        self.done = False
        self.edge_limit = 0
        self.event_queue = []  # This is our heap
        self.events = 0
        self.graph_name = 'data/graph1.txt'
        self.history_queue = []
        self.injections = 0
        self.next_destination = 0
        self.next_source = 0
        self.node_to_node = defaultdict(dict)
        self.send_fail = 0
        self.total_edges = 0
        self.total_edges_from_node = {}
        self.queue_full = 0
        self.total_hops = 0
        self.queue_limit = 100
        self.total_nodes = 0
        self.total_routing_time = 0.0
        self.routed_packets = 0
        self.rrh_connected_nodes = [0, 1, 2, 6, 7, 8]
        self.success_count = 0
        self.shortest = []
        self.viewer = None

    def _step(self, action):
        # if(self.total_routing_time/self.routed_packets < 10): #totally random, need change

        current_event = self.current_event
        current_time = current_event.event_time
        current_node = current_event.node

        # time_in_queue = current_time - current_event.q_time - self.internode

        # if the link wasn't good
        if action < 0 or action not in self.node_to_node[current_node]:
            # delete the Event
            resources_add_back = current_event.resources
            if resources_add_back:
                for i in resources_add_back:
                    src, act = i
                    l_num = self.absolute_node_edge_tuples[src][act]
                    self.resources_edges[l_num] += 1

            self.current_event = self.get_new_packet_bump()
            self.send_fail = self.send_fail + 1
            self.active_packets -= 1
            if self.current_event == NIL:
                return ((current_event.node, current_event.destination),
                        (current_event.node, current_event.destination)), self.done
            else:
                return ((current_event.node, current_event.destination),
                        (self.current_event.node, self.current_event.destination)), self.done

        else:
            next_node = self.node_to_node[current_node][action]
            l_num = self.absolute_node_edge_tuples[current_node][action]
            # Edge now occupied
            self.resources_edges[l_num] += -1
            current_event.hops += 1
            current_event.resources.append((current_node, action))
            # need to check link valid if in destination or not in destination
            # handle the case where next_node is your destination
            if next_node in self.bbu_connected_nodes:
                # cg: next node is one of bbu units
                # cg: check if there are enough resources at that destination
                # self.resources_edges[l_num]+=-1
                current_event.node = next_node  # do the send!
                self.resources_bbu[self.bbu_connected_nodes.index(next_node)] -= 1
                self.routed_packets += 1
                self.active_packets -= 1
                # ?! --> cg: need to add deletion to Event queue

                # ?! --> cg:add this completed route to history log
                current_event.destination = next_node
                current_event.q_time += 0.05
                current_event.q_time += 2.7
                self.history_queue.append((current_event.event_time, current_event.q_time))

                # ?! --> cg: add Event to system for when the item is suppose to leave # heap = ?
                heappush(
                    self.event_queue,
                    ((current_time + current_event.lifetime, -self.events), current_event)
                )

                self.current_event = self.get_new_packet_bump()

                if self.current_event == NIL:
                    return ((current_event.node, current_event.destination),
                            (current_event.node, current_event.destination)), self.done
                else:
                    return ((current_event.node, current_event.destination),
                            (self.current_event.node, self.current_event.destination)), self.done

            else:
                # cg: next node is not one of bbu units

                # self.resources_edges[l_num]+=-1
                current_event.node = next_node  # do the send!

                # add (source,action) to history of path

                current_event.q_time += 0.05
                next_time = current_event.event_time + .05
                current_event.event_time = next_time
                # self.enqueued[next_node] = next_time

                # current_event.q_time = current_time
                self.events += 1
                heappush(
                    self.event_queue,
                    ((current_time, -self.events), current_event)
                )
                self.current_event = self.get_new_packet_bump()

                if self.current_event == NIL:
                    return ((current_event.node, current_event.destination),
                            (current_event.node, current_event.destination)), self.done, {}
                else:
                    return ((current_event.node, current_event.destination),
                            (self.current_event.node, self.current_event.destination)), self.done

    def _reset(self):
        # self.distance, self.shortest = (zeros((0,0)),) * 2
        self.distance, self.shortest = (zeros((self.total_nodes, self.total_nodes)),) * 2
        self.done = False
        self.event_queue = []  # Q.PriorityQueue()
        self.events = 1
        self.history_queue = []
        self.read_in_graph()
        # bbu_connected dimensional array of bbu limit scalar
        self.resources_bbu = [self.bbu_limit] * len(self.bbu_connected_nodes)
        # total edge dimensional array of edge limit scalar
        self.resources_edges = [self.edge_limit] * self.total_edges
        self.send_fail = 0
        self.total_routing_time = 0.0

        # jl: for i in range(len(self.rrh_connected_nodes)):?
        for i in self.rrh_connected_nodes:
            self.events += 1
            self.injections += 1
            inject_event = Event(0.0, i)
            inject_event.source, inject_event.q_time = INJECT, 0.0
            # Call mean is the lambda parameter of the poisson distribution
            if self.call_mean == 1.0:
                inject_event.event_time = -m_log(random())
                # Why can we just call random? Dont we have to account for a continuous time?
            else:
                inject_event.event_time = -m_log(1 - random()) * float(self.call_mean)

            heappush(
                self.event_queue,
                ((inject_event.event_time, -self.events), inject_event)
            )

        self.current_event = self.get_new_packet_bump()

        return ((self.current_event.node, self.current_event.destination),
                (self.current_event.node, self.current_event.destination))

    # Helper functions

    # ?!--> is this comment reflective of the function? Initializes a signal at random node to a random destination
    def read_in_graph(self):
        self.total_nodes, self.total_edges = 0, 0
        graph_file = open(self.graph_name, "r")

        for line in graph_file:
            line_contents = line.split()

            if line_contents[0] == '1000':  # then the graph_file refers to a node
                self.total_edges_from_node[self.total_nodes] = 0
                self.total_nodes += 1

            if line_contents[0] == '2000':  # then the graph_file refers to connections between nodes
                node1, node2 = int(line_contents[1]), int(line_contents[2])

                self.node_to_node[node1][self.total_edges_from_node[node1]] = node2
                self.absolute_node_edge_tuples[node1][self.total_edges_from_node[node1]] = self.total_edges
                self.total_edges_from_node[node1] = self.total_edges_from_node[node1] + 1

                self.node_to_node[node2][self.total_edges_from_node[node2]] = node1
                self.absolute_node_edge_tuples[node2][self.total_edges_from_node[node2]] = self.total_edges
                self.total_edges_from_node[node2] = self.total_edges_from_node[node2] + 1

                self.total_edges += 1

    def reset_history(self):
        self.send_fail, self.history_queue = 0, []

    def start_packet(self, time, src):
        self.active_packets = self.active_packets + 1
        current_event = Event(time, src)
        current_event.source = src
        # current_event.source = current_event.node = source

        return current_event

    def get_new_packet_bump(self):
        # cg:needs to get updated
        current_event = heappop(self.event_queue)[1]
        while current_event.destination >= 0:
            resources_add_back = current_event.resources  # Add resources back

            for i in resources_add_back:
                src, act = i
                l_num = self.absolute_node_edge_tuples[src][act]
                self.resources_edges[l_num] += 1
            d = current_event.destination
            self.resources_bbu[self.bbu_connected_nodes.index(d)] += 1
            # get new item from queue
            current_event = heappop(self.event_queue)[1]
            # self.current_event = self.get_new_packet_bump()

        current_time = current_event.event_time
        src = current_event.node
        # make sure the Event we're sending the state of back is not an injection
        while current_event.source == INJECT:
            # cg: edit event_time of packet to decide when next packet of that type will enter system
            if self.call_mean == 1.0 or self.call_mean == 0.0:
                # jl: think poisson
                current_event.event_time += -m_log(1 - random())
            else:
                current_event.event_time += -m_log(1 - random()) * float(self.call_mean)

            # current_event.q_time = current_time  #cg:do i need this???
            current_event.q_time = 0
            heappush(
                self.event_queue,
                ((current_event.event_time, -self.events), current_event)
            )
            self.events += 1
            self.injections += 1
            # cg: packet enters system at this time
            current_event = self.start_packet(current_time, src)
            if current_event == NIL:
                current_event = heappop(self.event_queue)[1]

        if current_event == NIL:
            current_event = heappop(self.event_queue)[1]
        return current_event

    def calculate_reward(self):
        reward, l = 0, 0
        for i in self.history_queue:
            reward += -i[1]
            l += 1
        reward = reward - self.cost * self.send_fail
        l = l + self.send_fail
        avg_reward = reward / l
        return avg_reward

