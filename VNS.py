# Variable Neighbourhood Search
# author: Charles Lee
# date: 2022.12.10

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import trange, tqdm
import math
import common.Graph as GraphTools

# initial solution heuristic
class Solomon_Insertion():
    def __init__(self, graph):
        """
        solomon insertion algorithm to get an initial solution for VRP
        """
        self.name = "SolomonI1"
        """ set paraments """
        self.miu = 1
        self.lamda = 1 # ps: lambda is key word
        self.alpha1 = 1
        self.alpha2 = 0
        self.init_strategy = 3

        """ read data and preprocess """
        self.graph = graph
        default_vi = min(3, graph.vehicleTypeNum-1) # set default vi 
        self.weightCapacity = graph.weightCapacity[default_vi]
        self.volumnCapacity = graph.volumnCapacity[default_vi]

    def get_init_node(self, point_list):
        if self.init_strategy == 0: # 0: choose farthest
            max_d = 0
            for p in point_list:
                dist = self.graph.disMatrix[0, p]
                start_time = max(dist/self.graph.speed, self.graph.readyTime[p])
                if start_time > self.graph.dueTime[p]: # exclude point break time constraint
                    continue
                if dist > max_d:
                    max_d = dist
                    best_p = p # farthest point as max_pi
        elif self.init_strategy == 1: # 1: choose nearest
            min_d = np.inf
            for p in point_list:
                dist = self.graph.disMatrix[0, p]
                start_time = max(dist/self.graph.speed, self.graph.readyTime[p])
                if start_time > self.graph.dueTime[p]: # exclude point break time constraint
                    continue
                if dist < min_d:
                    min_d = dist
                    best_p = p # farthest point as max_pi
        elif self.init_strategy == 2: # 2: random select
            best_p = point_list[np.random.randint(len(point_list))]
        elif self.init_strategy == 3: # 3: highest due_time
            max_t = 0
            for p in point_list:
                due_time = self.graph.dueTime[p]
                start_time = max(self.graph.disMatrix[0, p]/self.graph.speed, self.graph.readyTime[p])
                if start_time > due_time: # exclude point break time constraint
                    continue
                if due_time > max_t:
                    max_t = due_time
                    best_p = p # farthest point as max_pi
        elif self.init_strategy == 4: # 4: highest start_time
            max_t = 0
            for p in point_list:
                due_time = self.graph.dueTime[p]
                start_time = max(self.graph.disMatrix[0, p]/self.graph.speed, self.graph.readyTime[p])
                if start_time > due_time: # exclude point break time constraint
                    continue
                if start_time > max_t:
                    max_t = start_time
                    best_p = p # farthest point as max_pi
        return best_p

    def run(self):
        """ construct a route each circulation """
        unassigned_points = list(range(1, self.graph.nodeNum)) 
        routes = []
        while len(unassigned_points) > 0: 
            # initiate load, point_list
            weight_load = 0
            volumn_load = 0
            point_list = unassigned_points.copy() # the candidate point set
            route_start_time_list = [0] # contains time when service started each point
            # choose the farthest point as s
            best_p = self.get_init_node(point_list)
            best_start_time = max(self.graph.disMatrix[0, best_p]/self.graph.speed, self.graph.readyTime[best_p])
            route = [0, best_p] # route contains depot and customer points 
            route_start_time_list.append(best_start_time) 
            point_list.remove(best_p) 
            unassigned_points.remove(best_p)
            weight_load += self.graph.weight[best_p]
            volumn_load += self.graph.volumn[best_p]

            """ add a point each circulation """
            while len(point_list) > 0:
                c2_list = [] # contains the best c1 value
                best_insert_list = [] # contains the best insert position
                # find the insert position with lowest additional distance
                pi = 0
                while pi < len(point_list):
                    u = point_list[pi]
                    # remove if over load
                    if weight_load + self.graph.weight[u] >= self.weightCapacity \
                       or volumn_load + self.graph.volumn[u] >= self.volumnCapacity: 
                        point_list.pop(pi)
                        continue
                    
                    best_c1 = np.inf 
                    for ri in range(len(route)):
                        i = route[ri]
                        if ri == len(route)-1:
                            rj = 0
                        else:
                            rj = ri+1
                        j = route[rj]
                        # c11 = diu + dui - miu*dij
                        c11 = self.graph.disMatrix[i, u] + self.graph.disMatrix[u, j] - self.miu * self.graph.disMatrix[i, j]
                        # c12 = bju - bj 
                        bj = route_start_time_list[rj]
                        bu = max(route_start_time_list[ri] + self.graph.serviceTime[i] + self.graph.disMatrix[i, u]/self.graph.speed, self.graph.readyTime[u])
                        bju = max(bu + self.graph.serviceTime[u] + self.graph.disMatrix[u, j]/self.graph.speed, self.graph.readyTime[j])
                        c12 = bju - bj

                        # remove if over time window
                        if bu > self.graph.dueTime[u] or bju > self.graph.dueTime[j]:
                            continue
                        PF = c12
                        pf_rj = rj
                        overtime_flag = 0
                        while PF > 0 and pf_rj < len(route)-1:
                            pf_rj += 1
                            bju = max(bju + self.graph.serviceTime[route[pf_rj-1]] + self.graph.disMatrix[route[pf_rj-1], route[pf_rj]], \
                                self.graph.readyTime[route[pf_rj]]) # start time of pf_rj
                            if bju > self.graph.dueTime[route[pf_rj]]:
                                overtime_flag = 1
                                break
                            PF = bju - route_start_time_list[pf_rj] # time delay
                        if overtime_flag == 1:
                            continue

                        # c1 = alpha1*c11(i,u,j) + alpha2*c12(i,u,j)
                        c1 = self.alpha1*c11 + self.alpha2*c12
                        # find the insert pos with best c1
                        if c1 < best_c1:
                            best_c1 = c1
                            best_insert = ri+1
                    # remove if over time (in all insert pos)
                    if best_c1 == np.inf:
                        point_list.pop(pi)
                        continue
                    c2 = self.lamda * self.graph.disMatrix[0, u] - best_c1
                    c2_list.append(c2)
                    best_insert_list.append(best_insert)
                    pi += 1
                if len(point_list) == 0:
                    break
                # choose the best point
                best_pi = np.argmax(c2_list)
                best_u = point_list[best_pi]
                best_u_insert = best_insert_list[best_pi] 
                # update route
                route.insert(best_u_insert, best_u)
                point_list.remove(best_u)
                unassigned_points.remove(best_u) # when point is assigned, remove from unassigned_points
                weight_load += self.graph.weight[best_u]
                volumn_load += self.graph.volumn[best_u]
                # update start_time
                start_time = max(route_start_time_list[best_u_insert-1] + self.graph.serviceTime[route[best_u_insert-1]] + self.graph.disMatrix[route[best_u_insert-1], best_u]/self.graph.speed, self.graph.readyTime[best_u])
                route_start_time_list.insert(best_u_insert, start_time)
                for ri in range(best_u_insert+1, len(route)):
                    start_time = max(route_start_time_list[ri-1] + self.graph.serviceTime[route[ri-1]] + self.graph.disMatrix[route[ri-1], route[ri]]/self.graph.speed, self.graph.readyTime[route[ri]])
                    route_start_time_list[ri] = start_time
            route.append(0)
            routes.append(route) 

        return routes

# neighbour stuctures (operators)
class Relocate():
    def __init__(self, k=1):
        self.k = k # how many points relocate together, k=1:relocate, k>1:Or-Opt

    def run(self, solution):
        """relocate point and the point next to it randomly inter/inner route (capacity not considered)

        Args:
            solution (List[int]): idxs of points of each route (route seperate with idx 0)

        Returns:
            neighbours (List[List[int]]): idxs of points of each route (seperate with idx 0) of each neighbour 
        """
        neighbours = []
        # 1. choose a point to relocate
        for pi in range(1, len(solution)-self.k):
            # 2. choose a position to put
            for li in range(1, len(solution)-self.k): # can't relocate to start/end
                neighbour = solution.copy()
                points = []
                for _ in range(self.k):
                    points.append(neighbour.pop(pi))
                for p in points[::-1]:
                    neighbour.insert(li, p)
                neighbours.append(neighbour)
        return neighbours     

    def get(self, solution):
        pi = np.random.randint(1, len(solution)-self.k)
        li = np.random.randint(1, len(solution)-self.k)
        neighbour = solution.copy()
        points = []
        for _ in range(self.k):
            points.append(neighbour.pop(pi))
        for p in points[::-1]:
            neighbour.insert(li, p)
        assert len(neighbour) == len(solution)
        return neighbour

class Exchange():
    def __init__(self, k=1):
        self.k = k # how many points exchange together

    def run(self, solution):
        """exchange two points randomly inter/inner route (capacity not considered)
        ps: Exchange operator won't change the points number of each vehicle

        Args:
            solution (List[int]): idxs of points of each route (route seperate with idx 0)

        Returns:
            neighbours (List[List[int]]): idxs of points of each route (seperate with idx 0) of each neighbour 
        """
        neighbours = []
        # 1. choose point i
        for pi in range(1, len(solution)-2*self.k-1):
            # 2. choose point j
            for pj in range(pi+self.k+1, len(solution)-self.k): 
                # if math.prod(solution[pi:pi+self.k]) == 0 or math.prod(solution[pj:pj+self.k]) == 0: # don't exchange 0
                #     continue
                neighbour = solution.copy()
                tmp = neighbour[pi:pi+self.k].copy()
                neighbour[pi:pi+self.k] = neighbour[pj:pj+self.k]
                neighbour[pj:pj+self.k] = tmp
                neighbours.append(neighbour)
        return neighbours    

    def get(self, solution):
        pi = np.random.randint(1, len(solution)-2*self.k-1)
        pj = np.random.randint(pi+self.k+1, len(solution)-self.k)
        while math.prod(solution[pi:pi+self.k]) == 0 or math.prod(solution[pj:pj+self.k]) == 0: # don't exchange 0
            pi = np.random.randint(1, len(solution)-2*self.k-1)
            pj = np.random.randint(pi+self.k+1, len(solution)-self.k)
        neighbour = solution.copy()
        tmp = neighbour[pi:pi+self.k].copy()
        neighbour[pi:pi+self.k] = neighbour[pj:pj+self.k]
        neighbour[pj:pj+self.k] = tmp
        assert len(neighbour) == len(solution)
        return neighbour

class Reverse():
    def __init__(self):
        pass

    def run(self, solution):
        """reverse route between two points randomly inter/inner route (capacity not considered)

        Args:
            solution (List[int]): idxs of points of each route (route seperate with idx 0)

        Returns:
            neighbours (List[List[int]]): idxs of points of each route (seperate with idx 0) of each neighbour 
        """
        neighbours = []
        # 1. choose point i
        for pi in range(1, len(solution)-2):
            # 2. choose point j
            for pj in range(pi+1, len(solution)-1): 
                neighbour = solution.copy()
                neighbour[pi:pj+1] = neighbour[pj:pi-1:-1]
                neighbours.append(neighbour)
        return neighbours 
    
    def get(self, solution):
        pi = np.random.randint(1, len(solution)-2)
        pj = np.random.randint(pi+1, len(solution)-1)
        neighbour = solution.copy()
        neighbour[pi:pj+1] = neighbour[pj:pi-1:-1]
        assert len(neighbour) == len(solution)
        return neighbour

# Variable Neighbourhood Search Algorithm
class VNS():
    def __init__(self, graph, iter_num=100000, heuristic=None):
        self.name = "VNS"
        self.graph = graph
        self.iter_num = iter_num 
        self.heuristic = heuristic
        # paraments for VNS
        self.operators_list = [Reverse(), Relocate(), Exchange()]
        # for k in range(3, 10):
        #     self.operators_list.append(Relocate(k))
        #     self.operators_list.append(Exchange(k))
        # display paraments
        self.process = []

    def solution_init(self):
        """
        generate initial solution randomly
        """
        if self.heuristic is not None:
            alg = self.heuristic(self.graph) 
            routes = alg.run()
            solution = [0]
            for route in routes:
                solution += route[1:]
        else:
            # all smallest vehicle
            vi = 0
            weight_capacity = self.graph.weightCapacity[vi]
            volumn_capacity = self.graph.volumnCapacity[vi]
            point_list = list(range(1, self.graph.nodeNum))
            np.random.shuffle(point_list)
            solution = [0]
            weight_sum = 0
            volumn_sum = 0
            for i in range(len(point_list)):
                pi = point_list[i]
                point_weight = self.graph.weight[pi]
                point_volumn = self.graph.volumn[pi]
                if weight_sum + point_weight < weight_capacity and \
                volumn_sum + point_volumn < volumn_capacity:
                    solution.append(pi)
                    weight_sum += point_weight
                    volumn_sum += point_volumn
                else:
                    solution.append(0)
                    solution.append(pi)
                    weight_sum = point_weight
                    volumn_sum = point_volumn
            solution.append(0) # add last 0
        self.best_solution = solution
        self.best_obj = self.cal_objective(solution)
        return solution
    
    def transfer(self, solution):
        """
        transfer solution to routes
        """
        routes = []
        for i, p in enumerate(solution[:-1]): # pass the end 0
            if p == 0:
                if i > 0:
                    routes[-1].append(0) # add end 0
                routes.append([0]) # add start 0
            else:
                routes[-1].append(p)
        else:
            routes[-1].append(0) # add final 0
        return routes
                
    def cal_objective(self, solution):
        """ calculate fitness(-obj) 
        obj = distance_cost + overload_cost + overtime_cost
        """
        routes = self.transfer(solution)
        obj = self.graph.cal_objective(routes) 
        return obj

    def get_neighbours(self, solution, operator):
        neighbours = operator.run(solution)
        return neighbours
    
    def choose_neighbour(self, neighbours):
        chosen_ni = np.random.randint(len(neighbours))
        return chosen_ni
        
    def remove_empty_vehicle(self, solution):
        idx = 1
        while idx < len(solution):
            if solution[idx-1] == 0 and solution[idx] == 0:
                solution.pop(idx)
            else:
                idx += 1
        return solution

    def show_result(self):
        self.best_routes = self.transfer(self.best_solution)
        print("Optimal Obj = {}".format(self.best_obj)) 
        vehicle_chosen = np.zeros(self.graph.vehicleTypeNum) # count number of each vehicle
        over_load_num = 0
        over_time_num = 0
        for ri in range(len(self.best_routes)):
            route = self.best_routes[ri]
            if len(route) <= 2:
                continue
            vi = self.graph.choose_vehicle(route)
            if vi is None:
                over_load_num += 1
            else:
                vehicle_chosen[vi] += 1
            tw_cost = self.graph.cal_time_window_cost(route)
            if tw_cost > 0:
                over_time_num += 1
        print("Vehicle:")
        for vi in range(self.graph.vehicleTypeNum):
            vehicleType = self.graph.vehicleTypeName[vi]
            vehicleNum = vehicle_chosen[vi]
            print("  {}: {}".format(vehicleType, vehicleNum))
        print("Overload {} routes, overtime {} routes".format(over_load_num, over_time_num))

    def show_process(self):
        y = self.process
        x = np.arange(len(y))
        plt.plot(x, y)
        plt.show() 

    def show_routes(self):
        self.best_routes = self.transfer(self.best_solution)
        self.graph.render(self.best_routes)

    def run(self):
        self.solution_init() # solution in form of routes
        neighbours = self.get_neighbours(self.best_solution, operator=self.operators_list[0])
        operator_k = 0
        for step in trange(self.iter_num):
            ni = self.choose_neighbour(neighbours)
            cur_solution = neighbours[ni]
            cur_solution = self.remove_empty_vehicle(cur_solution)
            cur_obj = self.cal_objective(cur_solution)
            # obj: minimize the total distance 
            if cur_obj < self.best_obj: 
                self.operators_list.insert(0, self.operators_list.pop(operator_k))
                operator_k = 0
                self.best_solution = cur_solution
                self.best_obj = cur_obj
                neighbours = self.get_neighbours(self.best_solution, operator=self.operators_list[0])
            else:
                neighbours.pop(ni)
                if len(neighbours) == 0: # when the neighbour space empty, change anothor neighbour structure(operator)
                    operator_k += 1
                    if operator_k < len(self.operators_list):
                        operator = self.operators_list[operator_k]
                        neighbours = self.get_neighbours(self.best_solution, operator=operator)
                    else:
                        print('local optimal, break out, iterated {} times'.format(step))
                        break

            self.process.append(self.best_obj)
        self.best_routes = self.transfer(self.best_solution)
        return self.best_routes

if __name__ == "__main__":
    graph = GraphTools.Graph()    
    alg = VNS(graph, iter_num=10000, heuristic=Solomon_Insertion)
    routes = alg.run()
    # alg.solution_init()
    alg.show_result() 
    alg.show_process()
    alg.show_routes()


    
                
                




