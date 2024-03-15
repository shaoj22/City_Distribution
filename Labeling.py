# Labeling Algorithm #todo
# author: Charles Lee
# date: 2022.12.15

import numpy as np
import matplotlib.pyplot as plt
import math
import common.Graph as GraphTools

class Label():
    def __init__(self, path, tabu, q, h, t, obj, dist, dual):
        self.path = path # current route
        self.tabu = tabu # can't visit point list
        self.q = q # current load of route
        self.h = h # current volumn of route
        self.t = t # start time of node
        self.obj = obj # the check number
        self.dist = dist # the total distance
        self.dual = dual # the total dual value
    
    @staticmethod
    def if_dominate(l1, l2):
        """check if l1 dominates l2 or on contrary

        Args:
            l1 (Label): label one 
            l2 (Label): label two
        Return:
            res (int): 0 stands for non-dominate, 1 for l1 dominate l2, 2 for l2 dominate l1
        """
        dominate_num = 0 
        if l1.obj < l2.obj:
            dominate_num += 1
        if l1.q < l2.q:
            dominate_num += 1
        if l1.h < l2.h:
            dominate_num += 1
        if l1.t < l2.t:
            dominate_num += 1
        
        cur_node = l1.path[-1]
        # if dominate_num == 4 and set(l1.path).issubset(l2.path): 
        if dominate_num == 4 and set(l1.tabu).issubset(l2.tabu): 
        # if dominate_num == 4:
            return 1
        # elif dominate_num == 0 and set(l2.path).issubset(l1.path): 
        elif dominate_num == 0 and set(l2.tabu).issubset(l1.tabu): 
        # elif dominate_num == 0:
            return 2
        else:
            return 0

class Labeling():
    def __init__(self, graph, select_num=1):
        self.name = "Labeling"
        self.graph = graph
        self.select_num = select_num # routes number generated at one time
        self.Q = [[] for i in range(self.graph.nodeNum)] # queue for each points, containing part-routes that ends in the point
        self.labelQueue = [] # label queue visit order of labels
        self.total_label_num = 0 # record
        self.total_dominant_num = 0 # record
        self.best_obj = np.inf # record
        self.EPS = 1e-5
        self.early_stop = 0 # stop if complete route num > 2 * select_num
        self.select_strategy = "DFS" # DFS / BFS / BestFS / random

        # node information
        self.nodeNum = self.graph.nodeNum
        self.dualValue = np.zeros(self.nodeNum) # initialize as 0
        self.location = self.graph.location
        self.weight = self.graph.weight
        self.volumn = self.graph.volumn
        self.disMatrix = self.graph.disMatrix
        self.readyTime = self.graph.readyTime
        self.dueTime = self.graph.dueTime
        self.serviceTime = self.graph.serviceTime
        # vehicle information
        self.vehicleTypeNum = self.graph.vehicleTypeNum
        self.speed = self.graph.speed
        self.weightCapacity = self.graph.weightCapacity
        self.volumnCapacity = self.graph.volumnCapacity
        self.startPrice = self.graph.startPrice
        self.meterPrice = self.graph.meterPrice
        
    def set_dual(self, Dual):
        self.dualValue = Dual
    
    def dominant_add(self, label, node):
        """
        add label to node, while checking dominance
        input:
            label (Label): label to add
            node (int): idx of the node
        update:
            self.Q (dict[int:List]): queue for each points
        """
        if self.early_stop and len(self.Q[0]) >= 2 * self.select_num:
            return 
        li = 0
        while li < len(self.Q[node]):
            labeli = self.Q[node][li]
            flag = Label.if_dominate(label, labeli)
            # if l1 dominates l2, pop(l2)
            if flag == 1:
                self.Q[node].pop(li)
                self.total_dominant_num += 1
            # if l2 dominates l1, not add l1
            elif flag == 2:
                self.total_dominant_num += 1
                return 
            li += 1
        self.Q[node].append(label)
        self.labelQueue.append(label)
        self.total_label_num += 1
    
    def label_select(self):
        if self.select_strategy == "DFS":
            return -1
        elif self.select_strategy == "BFS":
            return 0
        elif self.select_strategy == "BestFS":
            min_obj = np.inf
            best_li = None
            for li, label in enumerate(self.labelQueue):
                if label.obj < min_obj:
                    min_obj = label.obj 
                    best_li = li
            return best_li
        else: # random
            return np.random.randint(len(self.labelQueue))
            
    def label_expand(self, label):
        """
        expand each labels in the node
        input:
            label (Label): label to expand
        update:
            self.Q (dict[int:List]): queue of node 
        """
        node = label.path[-1] # node is the current point of label
        for next_node in self.graph.feasibleNodeSet[node]: # next_node: the next node
            if node == next_node: # avoid circulation
                continue
            if next_node in label.path[1:]: # avoid pass path, terminus 0 not included 
                continue
            q_ = label.q + self.weight[next_node]
            h_ = label.h + self.volumn[next_node]
            t_arrive = label.t + self.serviceTime[node] + self.disMatrix[node, next_node] / self.speed
            if q_ > max(self.weightCapacity) or h_ > max(self.volumnCapacity) or t_arrive > self.dueTime[next_node]: # check feasibility
                continue
            t_ = max(self.readyTime[next_node], t_arrive)
            # the correlation formula
            dist_ = label.dist + self.disMatrix[node, next_node]
            dual_ = label.dual + self.dualValue[next_node]
            w = self.choose_vehicle(q_, h_)
            obj_ = self.startPrice[w] + self.meterPrice[w] * dist_ - dual_ # cal obj according to vehicle
            if next_node == 0: # when route complete 
                if obj_ >=0: # only save negative obj route 
                    continue
                if obj_ < self.best_obj: # record best obj
                    self.best_obj = obj_
            path_ = label.path.copy()
            path_.append(next_node)
            tabu_ = label.tabu.copy()
            tabu_.append(next_node)
            tabu_ = list(set(tabu_ + self.graph.infeasibleNodeSet[next_node]))
            new_label = Label(path_, tabu_, q_, h_, t_, obj_, dist_, dual_)
            self.dominant_add(new_label, next_node) # add node and check dominance
    
    def choose_vehicle(self, q, h):
        for w in range(self.vehicleTypeNum):
            weight_capacity = self.weightCapacity[w]
            volumn_capacity = self.volumnCapacity[w]
            if weight_capacity > q and volumn_capacity > h:
                return w
        return None
    
    def select_best(self):
        pareto_labels = self.Q[0] 
        pareto_labels.sort(key=lambda label:label.obj)
        routes = [label.path for label in pareto_labels]
        objs = [label.obj for label in pareto_labels]
        return routes[:self.select_num], objs[:self.select_num]

    def run(self):
        label0 = Label([0], [], 0, 0, 0, 0, 0, 0)
        self.labelQueue.append(label0)
        self.total_label_num += 1
        self.label_expand(label0) 
        while len(self.labelQueue):
            # select and expand label
            li = self.label_select()
            label = self.labelQueue.pop(li)
            self.label_expand(label)
        routes, objs = self.select_best()
        return routes, objs
        
if __name__ == "__main__":
    graph = GraphTools.Graph()
    alg = Labeling(graph=graph, select_num=100)
    # Dual = [0] * alg.nodeNum
    Dual = np.arange(alg.nodeNum) * 100
    alg.set_dual(Dual)
    routes, objs = alg.run()
    for ri, route in enumerate(routes):
        print("{} obj: {}, route: {}".format(ri+1, objs[ri], route))
    print("total dominant num = {}".format(alg.total_dominant_num))
    
