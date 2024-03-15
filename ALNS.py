# Adaptive Large Neighbourhood Search
# author: Charles Lee
# date: 2022.12.13

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import math
from tqdm import trange, tqdm

import common.Graph as GraphTools
from VNS import VNS, Solomon_Insertion

class ReverseDestroy():
    def get(self, solution):
        new_solution = solution.copy()
        pi = np.random.randint(1, len(solution)-2)
        pj = np.random.randint(pi+1, len(solution)-1)
        new_solution[pi:pj+1] = new_solution[pj:pi-1:-1]
        destory_list = []
        return new_solution, destory_list

class RelocateDestroy():
    def __init__(self, k=1):
        self.k = k
    
    def get(self, solution):
        new_solution = solution.copy()
        pi = np.random.randint(1, len(solution)-self.k) 
        li = np.random.randint(1, len(solution)-self.k) 
        points = []
        for _ in range(self.k):
            points.append(new_solution.pop(pi))
        for p in points[::-1]:
            new_solution.insert(li, p)
        destory_list = []
        return new_solution, destory_list

class RandomDestory():
    def __init__(self, min_k = 2, max_k = 5):
        self.min_k = min_k # minimul destroy number
        self.max_k = max_k # maximum destroy number

    def get(self, solution):
        new_solution = solution.copy()
        # random destroy number
        destroy_num = min(np.random.randint(self.min_k, self.max_k), len(solution)-2)
        # randomly choose destroy place (can't destroy first/last 0)
        destroy_idx_list = np.random.choice(range(1, len(new_solution)-1), destroy_num, replace=False)
        destroy_list = [new_solution[i] for i in destroy_idx_list]
        # take destroy element out
        delete_count = 0
        idx = 0
        while idx < len(new_solution):
            check_idx = idx + delete_count
            if check_idx in destroy_idx_list:
                new_solution.pop(idx)
                delete_count += 1
            else:
                idx += 1
        return new_solution, destroy_list

class GreedyDestroy():
    def __init__(self, min_k=2, max_k=5, disMatrix=None):
        self.min_k = min_k # minimul destroy number
        self.max_k = max_k # maximum destroy number
        self.disMatrix = disMatrix # use to calculate obj

    def get(self, solution):
        new_solution = solution.copy()
        # random destroy number
        destroy_num = min(np.random.randint(self.min_k, self.max_k), len(solution)-2)
        # randomly choose destroy place (can't destroy first/last 0)
        destroy_list = []
        for i in range(destroy_num):
            max_extra_dist = -np.inf
            best_j = -1
            # find out the best place to destroy
            for j in range(1, len(new_solution)-1):
                p1 = new_solution[j-1]
                p2 = new_solution[j]
                p3 = new_solution[j+1]
                extra_dist = self.disMatrix[p1, p2] + self.disMatrix[p2, p3] - self.disMatrix[p1, p3]
                if extra_dist > max_extra_dist:
                    max_extra_dist = extra_dist
                    best_j = j
            if best_j == -1:
                best_j = np.random.randint(1, len(new_solution)-1)
            destroy_list.append(new_solution.pop(best_j))
        return new_solution, destroy_list       
                
class ShawDestroy():
    def __init__(self, min_k = 2, max_k = 5, disMatrix = None):
        self.min_k = min_k # minimul destroy number
        self.max_k = max_k # maximum destroy number
        self.disMatrix = disMatrix

    def get(self, solution):
        new_solution = solution.copy()
        # random destroy number
        destroy_num = min(np.random.randint(self.min_k, self.max_k), len(solution)-2)
        # randomly choose one destroy place (can't destroy first/last 0)
        first_destroy_idx = np.random.choice(range(1, len(new_solution)-1))
        dist_idx_list = []
        for idx in range(1, len(new_solution)-1):
            dist = self.disMatrix[new_solution[first_destroy_idx], new_solution[idx]]
            dist_idx_list.append([dist, idx])
        dist_idx_list.sort(key=lambda x:x[0])
        destroy_idx_list = [dist_idx_list[i][1] for i in range(len(dist_idx_list)) if i < destroy_num]
        destroy_list = [new_solution[i] for i in destroy_idx_list]
        # take destroy element out
        delete_count = 0
        idx = 0
        while idx < len(new_solution):
            check_idx = idx + delete_count
            if check_idx in destroy_idx_list:
                new_solution.pop(idx)
                delete_count += 1
            else:
                idx += 1
        return new_solution, destroy_list

class RandomRepair():
    def get(self, new_solution, destroy_list):
        for i in range(len(destroy_list)):
            insert_idx = np.random.randint(1, len(new_solution))
            new_solution.insert(insert_idx, destroy_list[i])
        return new_solution
        
class GreedyRepair():
    def __init__(self, disMatrix=None):
        self.disMatrix = disMatrix

    def get(self, new_solution, destroy_list):
        np.random.shuffle(destroy_list)
        for i in range(len(destroy_list)):
            min_extra_dist = np.inf
            for j in range(1, len(new_solution)):
                p1 = new_solution[j-1]
                p2 = destroy_list[i]
                p3 = new_solution[j]
                extra_dist = self.disMatrix[p1, p2] + self.disMatrix[p2, p3] - self.disMatrix[p1, p3]
                if extra_dist < min_extra_dist:
                    min_extra_dist = extra_dist
                    best_insert_j = j
            new_solution.insert(best_insert_j, destroy_list[i])
        return new_solution

class RegretRepair():
    def __init__(self, regret_n=8, disMatrix=None, alg=None):
        self.regret_n = regret_n
        self.disMatrix = disMatrix
        self.alg = alg
    
    def get(self, new_solution, destory_list):
        unassigned_list = destory_list.copy()
        routes = self.alg.transfer(new_solution)
        while unassigned_list:
            best_insert_list = []
            regret_list = []
            for pi in unassigned_list:
                extra_dist_list = []
                min_extra_dist = np.inf
                for j in range(1, len(new_solution)):
                    p1 = new_solution[j-1]
                    p2 = pi
                    p3 = new_solution[j]
                    extra_dist = self.disMatrix[p1, p2] + self.disMatrix[p2, p3] - self.disMatrix[p1, p3]
                    extra_dist_list.append(extra_dist)
                    if extra_dist < min_extra_dist:
                        min_extra_dist = extra_dist
                        best_insert_j = j
                extra_dist_list.sort() # increase order
                regret_list.append(sum(extra_dist_list[:self.regret_n]) - extra_dist_list[0]*self.regret_n)
                best_insert_list.append(best_insert_j)
            chosen_i = regret_list.index(max(regret_list))
            pi = unassigned_list.pop(chosen_i)
            best_insert = best_insert_list[chosen_i]
            new_solution.insert(best_insert, pi)
        return new_solution
                    
# Adaptive Large Neighbourhood Search
class ALNS(VNS):
    def __init__(self, graph, iter_num=None, heuristic=None):
        self.name = "ALNS"
        self.graph = graph
        if iter_num is not None:
            self.iter_num = iter_num
        else:
            self.iter_num = graph.nodeNum * graph.vehicleTypeNum * 200
        self.heuristic = heuristic
        # paraments for ALNS
        self.destroy_operators_list = [
                ReverseDestroy(),
                RelocateDestroy(), 
                RandomDestory(min_k=2, max_k=5), 
                GreedyDestroy(min_k=2, max_k=5, disMatrix=self.graph.disMatrix), 
                ShawDestroy(min_k=5, max_k=10, disMatrix=self.graph.disMatrix),
            ]
        self.repair_operators_list = [
                RandomRepair(), 
                GreedyRepair(self.graph.disMatrix),
                RegretRepair(8, self.graph.disMatrix, self),
            ]
        self.sigma1 = 33
        self.sigma2 = 9
        self.sigma3 = 13
        self.rho = 0.1
        self.destroy_operators_weights = np.ones(len(self.destroy_operators_list))
        self.destroy_operators_scores = np.ones(len(self.destroy_operators_list))
        self.destroy_operators_steps = np.ones(len(self.destroy_operators_list))
        self.repair_operators_weights = np.ones(len(self.repair_operators_list))
        self.repair_operators_scores = np.ones(len(self.repair_operators_list))
        self.repair_operators_steps = np.ones(len(self.repair_operators_list))
        # paraments for SA
        self.max_temp = 10
        self.min_temp = 0.1
        self.a = 0.97
        self.a_steps = 300
        # display paraments
        self.process = []

    def SA_accept(self, detaC, temperature):
        return math.exp(-detaC / temperature)

    def temperature_update(self, temperature):
        temperature *= self.a
        temperature = max(self.min_temp, temperature)
        return temperature

    def update_weights(self):
        # update weights
        destroy_deta_weights = self.destroy_operators_scores / self.destroy_operators_steps
        self.destroy_operators_weights = self.rho * self.destroy_operators_weights + (1 - self.rho) * destroy_deta_weights
        repair_deta_weights = self.repair_operators_scores / self.repair_operators_steps
        self.repair_operators_weights = self.rho * self.repair_operators_weights + (1 - self.rho) * repair_deta_weights
        # refresh scores / steps
        self.destroy_operators_scores = np.ones(len(self.destroy_operators_list))
        self.destroy_operators_steps = np.ones(len(self.destroy_operators_list))
        self.repair_operators_scores = np.ones(len(self.repair_operators_list))
        self.repair_operators_steps = np.ones(len(self.repair_operators_list))

    def choose_operator(self):
        # choose destroy operator
        prob1 = self.destroy_operators_weights / sum(self.destroy_operators_weights)
        opt_i1 = np.random.choice(range(len(self.destroy_operators_list)), p=prob1)
        # choose repair operator
        prob2 = self.repair_operators_weights / sum(self.repair_operators_weights)
        opt_i2 = np.random.choice(range(len(self.repair_operators_list)), p=prob2)
        return opt_i1, opt_i2

    def get_neighbour(self, solution, destroy_operator, repair_operator):
        new_solution, destroy_list = destroy_operator.get(solution)
        new_solution = repair_operator.get(new_solution, destroy_list)
        return new_solution

    def run(self):
        cur_solution = self.solution_init() # solution in form of routes
        cur_obj = self.cal_objective(cur_solution)
        self.best_solution = cur_solution
        self.best_obj = cur_obj
        temperature = self.max_temp
        for step in range(self.iter_num):
            opt_i1, opt_i2 = self.choose_operator()
            new_solution = self.get_neighbour(cur_solution, self.destroy_operators_list[opt_i1], self.repair_operators_list[opt_i2])
            new_solution = self.remove_empty_vehicle(new_solution) #? remove empty vehicle
            new_obj = self.cal_objective(new_solution)
            # obj: minimize the total distance 
            if new_obj < self.best_obj:
                self.best_solution = new_solution
                self.best_obj = new_obj
                cur_solution = new_solution
                cur_obj = new_obj
                self.destroy_operators_scores[opt_i1] += self.sigma1
                self.repair_operators_scores[opt_i2] += self.sigma1
            elif new_obj < cur_obj: 
                cur_solution = new_solution
                cur_obj = new_obj
                self.destroy_operators_scores[opt_i1] += self.sigma2
                self.repair_operators_scores[opt_i2] += self.sigma2
            elif np.random.random() < self.SA_accept(new_obj-cur_obj, temperature):
                cur_solution = new_solution
                cur_obj = new_obj
                self.destroy_operators_scores[opt_i1] += self.sigma3
                self.repair_operators_scores[opt_i2] += self.sigma3
            self.destroy_operators_steps[opt_i1] += 1
            self.repair_operators_steps[opt_i2] += 1
            # reset operators weights and update SA temperature
            if step % self.a_steps == 0: 
                self.update_weights()
                temperature = self.temperature_update(temperature)
            # record process obj
            self.process.append(cur_obj)
            if step % 100 == 0:
                print("iter {}, obj={}".format(step, cur_obj))
        self.best_routes = self.transfer(self.best_solution)
        return self.best_routes

    def visualize_route(self, coordinates, paths):
        path_coordinates = []
        for path in paths:
            path_coordinates.append([coordinates[idx] for idx in path])

        for path in path_coordinates:
            path_x, path_y = zip(*path)
            plt.plot(path_x, path_y, marker='o')

        warehouse_x, warehouse_y = coordinates[0]
        plt.plot(warehouse_x, warehouse_y, marker='s', color='red', markersize=10, label='仓库')

        plt.title('路径规划可视化')
        plt.xlabel('经度')
        plt.ylabel('纬度')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    graph = GraphTools.Graph()
    iter_num = 2000
    alg = ALNS(graph, iter_num, heuristic=Solomon_Insertion)
    routes = alg.run()
    print(routes)
    alg.show_result() 
    alg.show_process()
    # alg.show_routes()
    print("destroy weights: {}".format(alg.destroy_operators_weights))
    print("repair weights: {}".format(alg.repair_operators_weights))
    # graph.visualize_route(graph.location, routes)
    



