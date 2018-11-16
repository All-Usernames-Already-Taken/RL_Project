# based on code: https://github.com/Duncanswilson/q-routing-protocol
from abc import ABC
from collections import defaultdict
from heapq import heappush, heappop
from math import log as mlog
from random import random

import gym
from numpy import zeros

try:
    import Queue as Q  # ver. < 3.0
except ImportError:
    import queue as Q

events = 0

# /* Event structure. */


class Event:
    # Event is a packet?
    def __init__(self, time, src):
        self.dest, self.source = UNKNOWN, UNKNOWN
        self.birth, self.etime = time, time
        self.hops, self.qtime = 0, 0
        self.node = src

        # ?! --> cg: need to add status for lifetime in bbu site,
        # ?! -->need to add status for path taken, and which bbu resource used

        self.resources = []
        self.lifetime = 10


# /* Special events. */
INJECT, REPORT, END_SIM, UNKNOWN = -1, -2, -3, -4

# /* Define. */
NIL = Nil = -1


class NetworkSimulatorEnv(gym.Env, ABC):

    def __init__(self):

        self.viewer = None
        self.graph_name = 'data/graph1.txt'
        self.done = False
        self.total_routing_time = 0.0
        self.success_count = 0
        self.routed_packets = 0
        self.active_packets = 0
        self.send_fail = 0
        self.total_hops = 0
        self.total_nodes = 0
        self.total_edges = 0
        self.queue_limit = 100
        self.n_enqueued = {}
        self.enqueued = {}
        self.total_local_connections = {}
        self.inter_queue_n = []
        self.history_queue = []
        self.inter_node = 1.0
        self.inter_queue = 1.0
        self.history_queue = []
        self.event_queue = []  # Q.PriorityQueue()
        self.links = defaultdict(dict)
        self.abs_link_id = defaultdict(dict)
        self.current_event = Event(0.0, 0)  # do I need to do this?
        self.call_mean = 5  # Network load
        self.bbu_limit = 0
        self.edge_limit = 0
        self.cost = 0
        self.distance = []
        self.shortest = []
        self.next_destination = 0
        self.next_source = 0
        self.injections = 0
        self.queue_full = 0
        self.events = 0
        self.sources = [0, 1, 2, 6, 7, 8]  # Nodes connected to RRHs
        self.destinations = [3, 5]  # Nodes connected to BBU pools respectively

    def _step(self, action):
        # if(self.total_routing_time/self.routed_packets < 10): #totally random, need change

        current_event = self.current_event
        current_time = current_event.etime
        current_node = current_event.node

        # time_in_queue = current_time - current_event.qtime - self.internode

        # if the link wasnt good
        if action < 0 or action not in self.links[current_node]:
            # delete the Event
            resources_add_back = current_event.resources
            if resources_add_back:
                for i in resources_add_back:
                    src, act = i
                    l_num = self.abs_link_id[src][act]
                    self.resources_edges[l_num] += 1

            self.current_event = self.get_new_packet_bump()
            self.send_fail = self.send_fail + 1
            self.active_packets -= 1
            if self.current_event == NIL:
                return ((current_event.node, current_event.dest), (current_event.node, current_event.dest)), self.done
            else:
                return ((current_event.node, current_event.dest),
                        (self.current_event.node, self.current_event.dest)), self.done

        else:
            next_node = self.links[current_node][action]
            l_num = self.abs_link_id[current_node][action]
            self.resources_edges[l_num] += -1
            current_event.hops += 1
            current_event.resources.append((current_node, action))
            # need to check link valid if in dest or not in destination
            # handle the case where next_node is your destination
            if next_node in self.destinations:
                # cg: next node is one of bbu units
                # cg: check if there are enough resources at that destination
                # self.resources_edges[l_num]+=-1
                current_event.node = next_node  # do the send!
                self.resources_bbu[self.destinations.index(next_node)] -= 1
                self.routed_packets += 1
                self.active_packets -= 1
                # ?! --> cg: need to add deletion to Event queue

                # ?! --> cg:add this completed route to history log
                current_event.dest = next_node
                current_event.qtime += 0.05
                current_event.qtime += 2.7
                self.history_queue.append((current_event.etime, current_event.qtime))
                ###

                # ?! --> cg: add Event to system for when the item is suppose to leave
                heappush(self.event_queue, ((current_time + current_event.lifetime, -self.events), current_event))

                ##
                self.current_event = self.get_new_packet_bump()

                if self.current_event == NIL:
                    return ((current_event.node, current_event.dest),
                            (current_event.node, current_event.dest)), self.done
                else:
                    return ((current_event.node, current_event.dest),
                            (self.current_event.node, self.current_event.dest)), self.done

            else:
                # cg: next node is not one of bbu units

                # self.resources_edges[l_num]+=-1
                current_event.node = next_node  # do the send!

                # add (source,action) to history of path

                current_event.qtime += 0.05
                next_time = current_event.etime + .05
                current_event.etime = next_time
                # self.enqueued[next_node] = next_time

                # current_event.qtime = current_time
                self.events += 1
                heappush(self.event_queue, ((current_time, -self.events), current_event))
                self.current_event = self.get_new_packet_bump()

                if self.current_event == NIL:
                    return ((current_event.node, current_event.dest),
                            (current_event.node, current_event.dest)), self.done, {}
                else:
                    return ((current_event.node, current_event.dest),
                            (self.current_event.node, self.current_event.dest)), self.done

    def _reset(self):
        self.read_in_graph()
        self.distance = zeros((self.total_nodes, self.total_nodes))
        self.shortest = zeros((self.total_nodes, self.total_nodes))
        self.compute_best()
        self.done = False
        self.inter_queue_n = [self.inter_queue] * self.total_nodes

        self.event_queue = []  # Q.PriorityQueue()
        self.total_routing_time = 0.0

        self.enqueued = [0.0] * self.total_nodes
        self.n_enqueued = [0] * self.total_nodes
        self.send_fail = 0
        self.history_queue = []
        self.resources_edges = [self.edge_limit] * self.total_edges
        self.resources_bbu = [self.bbu_limit] * len(self.destinations)
        self.events = 1
        for i in self.sources:
            inject_event = Event(0.0, i)
            inject_event.source = INJECT
            # Call mean is the lambda parameter of the poisson distribution
            if self.call_mean == 1.0:
                inject_event.etime = -mlog(random())
            else:
                inject_event.etime = -mlog(1 - random()) * float(self.call_mean)

            inject_event.qtime = 0.0
            heappush(self.event_queue, ((inject_event.etime, -self.events), inject_event))
            self.injections += 1
            self.events += 1

        self.current_event = self.get_new_packet_bump()

        return ((self.current_event.node, self.current_event.dest), (self.current_event.node, self.current_event.dest))

    # Helper functions

    # Initializes a packet from a random source to a random destination
    def read_in_graph(self):
        self.total_nodes = 0
        self.total_edges = 0

        graph_file = open(self.graph_name, "r")

        for line in graph_file:
            line_contents = line.split()

            if line_contents[0] == '1000':  # node declaration

                self.total_local_connections[self.total_nodes] = 0
                self.total_nodes = self.total_nodes + 1

            if line_contents[0] == '2000':  # link declaration

                node1 = int(line_contents[1])
                node2 = int(line_contents[2])

                # link_num gives way to access global identifier for local links
                # links tells what nodes a node links to
                self.links[node1][self.total_local_connections[node1]] = node2
                self.abs_link_id[node1][self.total_local_connections[node1]] = self.total_edges
                self.total_local_connections[node1] = self.total_local_connections[node1] + 1

                self.links[node2][self.total_local_connections[node2]] = node1
                self.abs_link_id[node2][self.total_local_connections[node2]] = self.total_edges
                self.total_local_connections[node2] = self.total_local_connections[node2] + 1

                self.total_edges = self.total_edges + 1

    def reset_history(self):
        self.send_fail = 0
        self.history_queue = []

    def start_packet(self, time, src):

        self.active_packets = self.active_packets + 1
        current_event = Event(time, src)
        current_event.source = src
        # current_event.source = current_event.node = source

        return current_event

    def get_new_packet_bump(self):
        # cg:needs to get updated
        current_event = heappop(self.event_queue)[1]
        while current_event.dest >= 0:
            resources_add_back = current_event.resources  # Add resources back

            for i in resources_add_back:
                src, act = i
                l_num = self.abs_link_id[src][act]
                self.resources_edges[l_num] += 1
            d = current_event.dest
            self.resources_bbu[self.destinations.index(d)] += 1
            # get new item from queue
            current_event = heappop(self.event_queue)[1]
            # self.current_event = self.get_new_packet_bump()

        current_time = current_event.etime
        src = current_event.node
        # make sure the Event we're sending the state of back is not an injection
        while current_event.source == INJECT:
            # cg: edit e_time of packet to decide when next packet of that type will enter system
            if self.call_mean == 1.0 or self.call_mean == 0.0:
                current_event.etime += -mlog(1 - random())
            else:
                current_event.etime += -mlog(1 - random()) * float(self.call_mean)

            # current_event.qtime = current_time  #cg:do i need this???
            current_event.qtime = 0
            heappush(self.event_queue, ((current_event.etime, -self.events), current_event))
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
        reward = 0
        # what is l?
        l = 0
        for i in self.history_queue:
            reward += -i[1]
            l += 1
        reward = reward - self.cost * self.send_fail
        l = l + self.send_fail
        avg_reward = reward / l
        return avg_reward

    def add_link_lifetime_event(self):
        current_event.node = next_node  # Do the send!
        current_event.hops += 1
        current_event.source = -5
        # current_event
        # add (source,action) to history of path
        # current_event.hist
        heappush(self.event_queue, ((current_time + current_event.lifetime, -self.events), current_event))

    def compute_best(self):
        changing = True
        for i in range(self.total_nodes):
            for j in range(self.total_nodes):
                if i == j:
                    self.distance[i][j] = 0
                else:
                    self.distance[i][j] = self.total_nodes + 1
                self.shortest[i][j] = -1

        while changing:
            changing = False
            for i in range(self.total_nodes):
                for j in range(self.total_nodes):
                    # /* Update our estimate of distance for sending from i to j. */
                    if i != j:
                        for k in range(self.total_local_connections[i]):
                            if self.distance[i][j] > 1 + self.distance[self.links[i][k]][j]:
                                self.distance[i][j] = 1 + self.distance[self.links[i][k]][j]  # /* Better. */
                                self.shortest[i][j] = k
                                changing = True
