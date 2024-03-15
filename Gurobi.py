# Gurobi
# author: Charles Lee
# date: 2022.12.15

import numpy as np
import matplotlib.pyplot as plt
import math
import gurobipy as gp
import common.Graph as GraphTools

class GurobiSolver1():
    def __init__(self, graph, TimeLimit=300):
        self.name = "Gurobi"
        self.graph = graph
        self.TimeLimit = TimeLimit

    def build_MVVRPTW(self):
        """build MVVRPTW model

        Args:
            graph (Graph): contains all information needed in model, including:
                1. disMatrix of each node
                2. startPrice, meterPrice, weight_capacity, volumn_capacity, speed of each vehicle
                3. readyTime, dueTime, serviceTime of each node
        """
        # build model
        model = gp.Model("MVVRPTW")
        # add variables
        K = list(range(self.vehicleNum))
        N = list(range(self.graph.nodeNum))
        W = list(range(self.graph.vehicleTypeNum))
        x_list = [(i, j, k) for i in N for j in self.graph.feasibleNodeSet[i] for k in K]
        x = model.addVars(x_list, vtype="B", name="x") # k vehicle pass from i to j
        y_list = [(w, k) for w in W for k in K]
        y = model.addVars(y_list, vtype="B", name="y") # k vehicle use type w
        c_list = [k for k in K]
        c = model.addVars(c_list, vtype="C", name="c") # cost of vehicle k
        total_distance = model.addVars(c_list, vtype="C", name="td") # total distance of vehicle k
        t_list = [(i, k) for i in N for k in K]
        t = model.addVars(t_list, vtype="C", name="t") # arrive time of vehile k on node i
        # set objectives
        model.setObjective(gp.quicksum(c[k] for k in K), gp.GRB.MINIMIZE) # minimize total cost
        # add constraints
        # 1. cost part
        model.addConstrs((y[w, k] == 1) >> (c[k] >= self.graph.startPrice[w] + 
                                            self.graph.meterPrice[w] * total_distance[k])
                                            for w in W for k in K)
        model.addConstrs(total_distance[k] >= gp.quicksum(self.graph.disMatrix[i, j] * x[i, j, k] for i in N for j in self.graph.feasibleNodeSet[i]) for k in K)
        model.addConstrs(gp.quicksum(y[w, k] for w in W) <= 1 for k in K)
        # # 2. route part
        model.addConstrs(gp.quicksum(x[i, j, k] for i in self.graph.availableNodeSet[j]) == gp.quicksum(x[j, i, k] for i in self.graph.feasibleNodeSet[j]) for j in N for k in K)
        model.addConstrs(gp.quicksum(x[i, 0, k] for i in self.graph.availableNodeSet[0]) == gp.quicksum(y[w, k] for w in W) for k in K)
        model.addConstrs(gp.quicksum(x[i, j, k] for i in self.graph.availableNodeSet[j] for k in K) == 1 for j in N[1:])
        # # 3. weight part
        model.addConstrs(gp.quicksum(x[i, j, k] * self.graph.weight[i] for i in N for j in self.graph.feasibleNodeSet[i]) 
                         <= gp.quicksum(y[w, k] * self.graph.weightCapacity[w] for w in W) for k in K)
        # 4. volumn part
        model.addConstrs(gp.quicksum(x[i, j, k] * self.graph.volumn[i] for i in N for j in self.graph.feasibleNodeSet[i]) 
                         <= gp.quicksum(y[w, k] * self.graph.volumnCapacity[w] for w in W) for k in K)
        # 5. time part
        model.addConstrs((x[i, j, k] == 1) >> (t[j, k] >= t[i, k] + self.graph.serviceTime[i] + self.graph.disMatrix[i, j] / self.graph.speed)
                                    for k in K for i in N for j in self.graph.feasibleNodeSet[i] if j != 0)
        model.addConstrs(t[i, k] >= self.graph.readyTime[i] for i in N for k in K)
        model.addConstrs(t[i, k] <= self.graph.dueTime[i] for i in N for k in K)

        # update model
        model.update()
        
        return model
        
    def solve_and_get_routes(self):
        # solve model 
        self.model.optimize()
        # get routes
        K = list(range(self.vehicleNum))
        N = list(range(self.graph.nodeNum))
        W = list(range(self.graph.vehicleTypeNum))
        routes = []
        for k in K:
            route = [0] 
            cur_i = 0
            while 1:
                next_j = None
                for j in self.graph.feasibleNodeSet[cur_i]:
                    if j == cur_i:
                        continue
                    var_name = "x[{},{},{}]".format(cur_i, j, k)
                    var = self.model.getVarByName(var_name)
                    if abs(var.X - 1) < 1e-3:
                        next_j = j
                        break
                if next_j is None and cur_i == 0: # empty vehicle
                    break
                assert next_j is not None, "Model Error: Dead end route"
                route.append(next_j)
                cur_i = next_j
                if next_j == 0:
                    break
            if len(route) <= 1:
                continue
            routes.append(route)
        return routes

    def show_routes(self):
        self.graph.render(self.routes) 

    def show_results(self):
        # print("Optimal objective = {}".format(self.model.ObjVal)) 
        print("Optimal objective = {}".format(self.graph.cal_objective(self.routes))) 
        print("Vehicle:")
        for w in range(self.graph.vehicleTypeNum):
            vehicle_num = 0
            for k in range(self.vehicleNum):
                var_name = "y[{},{}]".format(w, k)
                var = self.model.getVarByName(var_name)
                vehicle_num += round(var.X)
            print("  {}: {}".format(self.graph.vehicleTypeName[w], vehicle_num))
    
    def run(self):
        self.vehicleNum = 15 # set max vehicle number
        self.model = self.build_MVVRPTW()
        if self.TimeLimit is not None:
            self.model.setParam("TimeLimit", self.TimeLimit)
        self.routes = self.solve_and_get_routes()
        return self.routes

class GurobiSolver2():
    def __init__(self, graph, TimeLimit=100):
        self.name = "Gurobi"
        self.graph = graph
        self.TimeLimit = TimeLimit

    def build_MVVRPTW(self):
        """build MVVRPTW model

        Args:
            graph (Graph): contains all information needed in model, including:
                1. disMatrix of each node
                2. startPrice, meterPrice, weight_capacity, volumn_capacity, speed of each vehicle
                3. readyTime, dueTime, serviceTime of each node
        """
        # build model
        model = gp.Model("MVVRPTW")
        # add variables
        N = list(range(self.graph.nodeNum))
        W = list(range(self.graph.vehicleTypeNum))
        x_list = [(i, j, w) for i in N for j in self.graph.feasibleNodeSet[i] for w in W]
        x = model.addVars(x_list, vtype="B", name="x") # k vehicle pass from i to j
        q_list = [(i, w) for i in N for w in W]
        q = model.addVars(q_list, vtype="C", name="q") # weight load of vehile type w on node i
        h = model.addVars(q_list, vtype="C", name="h") # volumn load of vehile type w on node i
        t = model.addVars(q_list, vtype="C", name="t") # arrive time of vehile type w on node i
        # set objectives
        model.setObjective(gp.quicksum(self.graph.startPrice[w] * gp.quicksum(x[0, i, w] for i in self.graph.feasibleNodeSet[0])
                                        + self.graph.meterPrice[w] * gp.quicksum(x[i, j, w] * self.graph.disMatrix[i, j] for i in N for j in self.graph.feasibleNodeSet[i])
                                        for w in W), gp.GRB.MINIMIZE) # minimize total cost
        # add constraints
        # 1. finish all orders
        model.addConstrs(gp.quicksum(x[i, j, w] for j in self.graph.feasibleNodeSet[i] for w in W) >= 1 for i in N)
        # 2. flow balance
        model.addConstrs(gp.quicksum(x[j, i, w] for j in self.graph.availableNodeSet[i]) == gp.quicksum(x[i, j, w] for j in self.graph.feasibleNodeSet[i]) for i in N for w in W)
        # 3. weight constraints
        model.addConstrs((x[i, j, w] == 1) >> (q[j, w] >= q[i, w] + self.graph.weight[j]) for i in N for j in self.graph.feasibleNodeSet[i] if j!=0 for w in W)
        model.addConstrs(q[i, w] <= self.graph.weightCapacity[w] for i in N for w in W)
        model.addConstrs(q[i, w] >= 0 for i in N for w in W)
        # 4. volumn constraints
        model.addConstrs((x[i, j, w] == 1) >> (h[j, w] >= h[i, w] + self.graph.volumn[j]) for i in N for j in self.graph.feasibleNodeSet[i] if j!=0 for w in W)
        model.addConstrs(h[i, w] <= self.graph.volumnCapacity[w] for i in N for w in W)
        model.addConstrs(h[i, w] >= 0 for i in N for w in W)
        # 5. time window
        model.addConstrs((x[i, j, w] == 1) >> (t[j, w] >= t[i, w] + self.graph.serviceTime[i] + self.graph.disMatrix[i, j]/self.graph.speed) for i in N for j in self.graph.feasibleNodeSet[i] if j!=0 for w in W)
        model.addConstrs(t[i, w] <= self.graph.dueTime[i] for i in N for w in W)
        model.addConstrs(t[i, w] >= self.graph.readyTime[i] for i in N for w in W)

        # update model
        model.update()

        # preprocess
        for i in N:
            for w in W:
                if (self.graph.weight[i] > self.graph.weightCapacity[w] 
                    or self.graph.volumn[i] > self.graph.volumnCapacity[w]):
                    for j in self.graph.feasibleNodeSet[i]:
                        var_name = f"x[{i},{j},{w}]"
                        model.getVarByName(var_name).setAttr("UB", 0)
                    for j in self.graph.availableNodeSet[i]:
                        var_name = f"x[{j},{i},{w}]"
                        model.getVarByName(var_name).setAttr("UB", 0)
                        
        return model
        
    def get_routes(self):
        # get routes
        N = list(range(self.graph.nodeNum))
        W = list(range(self.graph.vehicleTypeNum))
        routes = []
        for w in W:
            # get first point of each routes
            first_points = []
            for j in self.graph.feasibleNodeSet[0]:
                var_name = "x[{},{},{}]".format(0, j, w)
                var = self.model.getVarByName(var_name)
                if abs(var.X - 1) < 1e-3:
                    first_points.append(j)
            # get route start from first point
            for first_point in first_points:
                route = [0] + [first_point]
                cur_i = first_point
                while 1:
                    next_j = None
                    for j in self.graph.feasibleNodeSet[cur_i]:
                        if j == cur_i:
                            continue
                        var_name = "x[{},{},{}]".format(cur_i, j, w)
                        var = self.model.getVarByName(var_name)
                        if abs(var.X - 1) < 1e-3:
                            next_j = j
                            break
                    if next_j is None and cur_i == 0: # empty vehicle
                        break
                    assert next_j is not None, "Model Error: Dead end route"
                    route.append(next_j)
                    cur_i = next_j
                    if next_j == 0:
                        break
                if len(route) <= 1:
                    continue
                routes.append(route)
        return routes

    def show_routes(self):
        self.graph.render(self.routes) 

    def show_results(self):
        print("Optimal objective = {}".format(self.model.ObjVal)) 
        print("Vehicle:")
        for w in range(self.graph.vehicleTypeNum):
            vehicle_num = 0
            for j in self.graph.feasibleNodeSet[0]:
                var_name = "x[{},{},{}]".format(0, j, w)
                var = self.model.getVarByName(var_name)
                vehicle_num += round(var.X)
            print("  {}: {}".format(self.graph.vehicleTypeName[w], vehicle_num))
    
    def run(self):
        self.model = self.build_MVVRPTW()
        if self.TimeLimit is not None:
            self.model.setParam("TimeLimit", self.TimeLimit)
        self.model.optimize()
        self.routes = self.get_routes()
        return self.routes

if __name__ == "__main__":
    graph = GraphTools.Graph()
    solver = GurobiSolver1(graph, TimeLimit=None)
    solver.run()
    solver.show_results()
    solver.show_routes()
        


