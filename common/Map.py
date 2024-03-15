# Map class

import numpy as np
import matplotlib.pyplot as plt

from Node import Node

class Map:
    def __init__(self, node_list, disMatrix):
        self.node_list = node_list
        self.disMatrix = disMatrix
    
    def get_index_of_place(self, place_str):
        for ni in range(len(self.node_list)):
            if self.node_list[ni].address == place_str:
                return ni
        assert 0, "place not exists"
    
    def render(self, node_list=None, routes=[]):
        if node_list is None:
            node_list = self.node_list
        # show all nodes
        X_list1 = []
        Y_list1 = []
        for ni in range(len(node_list)):
            X_list1.append(node_list[ni].posX)
            Y_list1.append(node_list[ni].posY)
        plt.scatter(X_list1, Y_list1, c='black')
        # show vehicles
        X_list2 = []
        Y_list2 = []
        for vi in range(len(routes)):
            ni = routes[vi][0] # first node is vehicle node
            X_list2.append(node_list[ni].posX)
            Y_list2.append(node_list[ni].posY)
        plt.scatter(X_list2, Y_list2, c='red', s=50, marker='^')
        # show routes
        for vi in range(len(routes)):
            route_X = []
            route_Y = []
            for ni in routes[vi]:
                route_X.append(node_list[ni].posX)
                route_Y.append(node_list[ni].posY)
            plt.plot(route_X, route_Y)
        # show whole picture
        plt.show()


        
            
    