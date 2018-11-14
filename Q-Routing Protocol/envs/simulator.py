import gym
import numpy as np
import heapq
import collections
from os import path
from os import sys
import math
import random

try:
    import Queue as Q  # ver. < 3.0
except ImportError:
    import queue as Q

events = 0


# based on code: https://github.com/Duncanswilson/q-routing-protocol
# /* Event structure. */
class event:
    def __init__(self, time, src):
        # /* Initialize new event. */
        self.dest = UNKNOWN
        self.source = UNKNOWN
        self.node = src
        self.birth = time
        self.hops = 0
        self.etime = time
        self.qtime = 0
        # cg: need to add status for lifetime in bbu site,
        # need to add status for path taken, and which bbu resource used
        self.resources = []
        self.lifetime = 10


# /* Special events. */
INJECT = -1
REPORT = -2
END_SIM = -3
UNKNOWN = -4

# /* Define. */
NIL = Nil = -1


class NetworkSimulatorEnv(gym.Env):

    # We init the network simulator here
    def __init__(self):
        self.viewer = None
        self.graphname = 'data/graph1.txt'
        self.done = False
        self.success_count = 0
        self.nnodes = 0
        self.nedges = 0
        self.enqueued = {}
        self.nenqueued = {}
        self.interqueuen = []
        self.event_queue = []  # Q.PriorityQueue()
        self.history_queue = []
        self.nlinks = {}
        self.links = collections.defaultdict(dict)
        self.link_num = collections.defaultdict(dict)
        self.total_routing_time = 0.0
        self.routed_packets = 0
        self.total_hops = 0
        self.current_event = event(0.0, 0)  # do I need to do this?
        self.internode = 1.0
        self.interqueue = 1.0
        self.active_packets = 0
        self.queuelimit = 100
        self.send_fail = 0
        self.history_queue = []
        #
        self.callmean = 5  # network load
        self.bbu_limit = 0
        self.edge_limit = 0
        self.cost = 0

        self.distance = []  # np.zeros((self.nnodes,self.nnodes))
        self.shortest = []  # np.zeros((self.nnodes,self.nnodes))

        self.next_dest = 0
        self.next_source = 0
        self.injections = 0
        self.queue_full = 0

        self.events = 0

        self.sources = [0, 1, 2, 6, 7, 8]
        self.dests = [3, 5]
        self.next_source = 0
        self.next_dest = 0

    def _step(self, action):
        # if(self.total_routing_time/self.routed_packets < 10): #totally random, need change

        current_event = self.current_event
        current_time = current_event.etime
        current_node = current_event.node

        # time_in_queue = current_time - current_event.qtime - self.internode

        # if the link wasnt good
        if action < 0 or action not in self.links[current_node]:
            # delete the event
            resources_add_back = current_event.resources
            if resources_add_back:
                for i in resources_add_back:
                    src, act = i
                    l_num = self.link_num[src][act]
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
            l_num = self.link_num[current_node][action]
            self.resources_edges[l_num] += -1
            current_event.hops += 1
            current_event.resources.append((current_node, action))
            # need to check link valid if in dest or not in destination
            # handle the case where next_node is your destination
            if next_node in self.dests:
                # cg: next node is one of bbu units
                # cg: check if there are enough resources at that destination
                # self.resources_edges[l_num]+=-1
                current_event.node = next_node  # do the send!
                self.resources_bbu[self.dests.index(next_node)] -= 1
                self.routed_packets += 1
                self.active_packets -= 1
                # cg: need to add deletion to event queue

                ###cg:add this completed route to history log
                current_event.dest = next_node
                current_event.qtime += 0.05
                current_event.qtime += 2.7
                self.history_queue.append((current_event.etime, current_event.qtime))
                ###

                ##cg: add event to system for when the item is suppose to leave
                heapq.heappush(self.event_queue, ((current_time + current_event.lifetime, -self.events), current_event))

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
                heapq.heappush(self.event_queue, ((current_time, -self.events), current_event))
                self.current_event = self.get_new_packet_bump()

                if self.current_event == NIL:
                    return ((current_event.node, current_event.dest),
                            (current_event.node, current_event.dest)), self.done, {}
                else:
                    return ((current_event.node, current_event.dest),
                            (self.current_event.node, self.current_event.dest)), self.done

    def _reset(self):
        self.readin_graph()
        self.distance = np.zeros((self.nnodes, self.nnodes))
        self.shortest = np.zeros((self.nnodes, self.nnodes))
        self.compute_best()
        self.done = False
        self.interqueuen = [self.interqueue] * self.nnodes

        self.event_queue = []  # Q.PriorityQueue()
        self.total_routing_time = 0.0

        self.enqueued = [0.0] * self.nnodes
        self.nenqueued = [0] * self.nnodes
        self.send_fail = 0
        self.history_queue = []
        self.resources_edges = [self.edge_limit] * self.nedges
        self.resources_bbu = [self.bbu_limit] * len(self.dests)
        self.events = 1
        for i in self.sources:
            inject_event = event(0.0, i)
            inject_event.source = INJECT
            if self.callmean == 1.0:
                inject_event.etime = -math.log(random.random())
            else:
                inject_event.etime = -math.log(1 - random.random()) * float(self.callmean)

            inject_event.qtime = 0.0
            heapq.heappush(self.event_queue, ((inject_event.etime, -self.events), inject_event))
            self.injections += 1
            self.events += 1

        self.current_event = self.get_new_packet_bump()

        return ((self.current_event.node, self.current_event.dest), (self.current_event.node, self.current_event.dest))

    ###########helper functions############################
    # Initializes a packet from a random source to a random destination
    def readin_graph(self):
        self.nnodes = 0
        self.nedges = 0

        graph_file = open(self.graphname, "r")

        for line in graph_file:
            line_contents = line.split()

            if line_contents[0] == '1000':  # node declaration

                self.nlinks[self.nnodes] = 0
                self.nnodes = self.nnodes + 1

            if line_contents[0] == '2000':  # link declaration

                node1 = int(line_contents[1])
                node2 = int(line_contents[2])

                # link_num gives way to access global identifier for local links
                # nlinks is total number of links for a given node
                # links tells what nodes a node links to
                self.links[node1][self.nlinks[node1]] = node2
                self.link_num[node1][self.nlinks[node1]] = self.nedges
                self.nlinks[node1] = self.nlinks[node1] + 1

                self.links[node2][self.nlinks[node2]] = node1
                self.link_num[node2][self.nlinks[node2]] = self.nedges
                self.nlinks[node2] = self.nlinks[node2] + 1

                self.nedges = self.nedges + 1

    def reset_history(self):
        self.send_fail = 0
        self.history_queue = []

    def start_packet(self, time, src):

        self.active_packets = self.active_packets + 1
        current_event = event(time, src)
        current_event.source = src
        # current_event.source = current_event.node = source

        return current_event

    def get_new_packet_bump(self):
        # cg:needs to get updated
        current_event = heapq.heappop(self.event_queue)[1]
        while current_event.dest >= 0:
            # add resources back
            resources_add_back = current_event.resources
            for i in resources_add_back:
                src, act = i
                l_num = self.link_num[src][act]
                self.resources_edges[l_num] += 1
            d = current_event.dest
            self.resources_bbu[self.dests.index(d)] += 1
            # get new item from queue
            current_event = heapq.heappop(self.event_queue)[1]
            # self.current_event = self.get_new_packet_bump()

        current_time = current_event.etime
        src = current_event.node
        # make sure the event we're sending the state of back is not an injection
        while current_event.source == INJECT:
            # cg: edit e_time of packet to decide when next packet of that type will enter system
            if self.callmean == 1.0 or self.callmean == 0.0:
                current_event.etime += -math.log(1 - random.random())
            else:
                current_event.etime += -math.log(1 - random.random()) * float(self.callmean)

            # current_event.qtime = current_time  #cg:do i need this???
            current_event.qtime = 0
            heapq.heappush(self.event_queue, ((current_event.etime, -self.events), current_event))
            self.events += 1
            self.injections += 1
            # cg: packet enters system at this time
            current_event = self.start_packet(current_time, src)
            if current_event == NIL:
                current_event = heapq.heappop(self.event_queue)[1]

        if current_event == NIL:
            current_event = heapq.heappop(self.event_queue)[1]
        return current_event

    def calculate_reward(self):
        reward = 0
        l = 0
        for i in self.history_queue:
            reward += -i[1]
            l += 1
        reward = reward - self.cost * self.send_fail
        l = l + self.send_fail
        avg_reward = reward / l
        return avg_reward

    def add_link_lifetime_event(self):
        current_event.node = next_node  # do the send!
        current_event.hops += 1
        current_event.source = -5
        # current_event
        # add (source,action) to history of path
        # current_event.hist
        heapq.heappush(self.event_queue, ((current_time + current_event.lifetime, -self.events), current_event))

    def compute_best(self):
        changing = True
        for i in range(self.nnodes):
            for j in range(self.nnodes):
                if i == j:
                    self.distance[i][j] = 0
                else:
                    self.distance[i][j] = self.nnodes + 1
                self.shortest[i][j] = -1

        while changing:
            changing = False
            for i in range(self.nnodes):
                for j in range(self.nnodes):
                    # /* Update our estimate of distance for sending from i to j. */
                    if i != j:
                        for k in range(self.nlinks[i]):
                            if self.distance[i][j] > 1 + self.distance[self.links[i][k]][j]:
                                self.distance[i][j] = 1 + self.distance[self.links[i][k]][j]  # /* Better. */
                                self.shortest[i][j] = k
                                changing = True
