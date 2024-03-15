# Branch and Price #todo
# author: Charles Lee
# date: 2022.12.15

import numpy as np
import matplotlib.pyplot as plt
import math
import time
from copy import deepcopy
import gurobipy as gp
from gurobipy import GRB
from Labeling import Labeling
import common.Graph as GraphTools

class BPNode():
    def __init__(self, graph, RLMP, SP, global_column_pool):
        self.graph = graph
        self.global_column_pool = global_column_pool
        self.RLMP = RLMP.copy()
        self.SP = SP
        # model part
        self.duals_of_RLMP = {}
        self.SP_must_include = []
        self.SP_cant_include = []
        self.max_not_improve_cnt = 10
        self.select_num = 50
        # algorithm part
        self.local_LB = np.inf
        self.IP_obj = np.inf
        self.LP_obj = -np.inf
        self.EPS = 1e-6 
        self.is_feasible = False
        self.is_integer = False
        self.inf_var_list = []
        self.x_sol = {}
        self.x_int_sol = {}
        # display part
        self.cg_iter_cnt = 0
        self.depth = 0
        self.has_showed_way_of_opt = False
        self.way_of_opt = "---"
        self.prune_info = "---"
    
    def generate(self):
        subNode = BPNode(self.graph, self.RLMP, self.SP, self.global_column_pool)
        subNode.local_LB = self.LP_obj
        subNode.depth = self.depth + 1
        subNode.SP_must_include = self.SP_must_include.copy()
        subNode.SP_cant_include = self.SP_cant_include.copy()
        return subNode
    
    def solve_and_update(self):
        """ solve and check feasibility """
        self.is_feasible = self.column_generation_labeling()
        if self.is_feasible == 0:
            return
        """ update x_sol and round to x_int_sol """
        vars = self.RLMP.getVars()
        self.is_integer = True
        for var in vars:
            var_name = var.VarName
            var_val = var.X
            self.x_sol[var_name] = var_val
            self.x_int_sol[var_name] = round(var_val)
            # check integer
            if (abs(round(var_val) - var_val) > self.EPS):
                self.is_integer = False
                self.inf_var_list.append(var_name)
        """ update LP / IP """
        self.LP_obj = self.RLMP.ObjVal
        if self.is_integer:
            self.way_of_opt = "By Simplex"
            obj = 0
            for var in vars:
                var_name = var.VarName
                obj += self.x_int_sol[var_name] * var.obj
            self.IP_obj = obj
        else:
            # solve RMP to get a integer solution as UB
            self.solve_final_RMP_and_update_IPobj()
    
    def column_generation(self):
        self.set_SP()
        best_RLMP_obj = np.inf
        not_improve_cnt = 0
        while True:
            # solve RLMP and get duals
            is_feasible = self.solve_RLMP_and_get_duals()
            if is_feasible == 0:
                return 0 # node infeasible
            if self.RLMP.ObjVal < best_RLMP_obj:
                not_improve_cnt = 0
                best_RLMP_obj = self.RLMP.ObjVal
                self.SP.setParam("PoolSolutions", 1) 
            else:
                not_improve_cnt += 1
                if not_improve_cnt > self.max_not_improve_cnt:
                    self.SP.setParam("PoolSolutions", not_improve_cnt//10) 
            # solve SP
            self.solve_SP()
            if self.SP.Status != 2:
                return 0 # node infeasible
            # break if can't find route to improve RLMP
            if self.SP.ObjVal >= -self.EPS:
                return 1 # node feasible
            # get columns and add into RLMP
            self.get_columns_from_SP_and_add()
            self.cg_iter_cnt += 1
        self.reset_SP()
     
    def column_generation_labeling(self):
        best_RLMP_obj = np.inf
        not_improve_cnt = 0
        while True:
            # solve RLMP and get duals
            is_feasible = self.solve_RLMP_and_get_duals()
            if is_feasible == 0:
                return 0 # node infeasible
            # get routes from labeling and add to RLMP
            min_obj = self.get_columns_from_Labeling_and_add()
            # break if can't find route to improve RLMP
            if min_obj >= -self.EPS:
                return 1 # node feasible
            self.cg_iter_cnt += 1
    
    def solve_RLMP_and_get_duals(self):
        self.RLMP.optimize()
        if self.RLMP.Status != 2:
            return 0 # RLMP infeasible
        for cons in self.RLMP.getConstrs():
            cons_name = cons.ConstrName 
            self.duals_of_RLMP[cons_name] = cons.Pi
        return 1 # RLMP feasible
    
    def solve_SP(self):
        # update objective with duals
        obj = 0.0
        for i in range(self.graph.nodeNum):
            for j in self.graph.feasibleNodeSet[i]:
                var_name = f"x[{i},{j}]"
                cons_name = f"R{i}"
                coef = self.graph.disMatrix[i, j] - self.duals_of_RLMP[cons_name]
                obj += coef * self.SP.getVarByName(var_name)
        self.SP.setObjective(obj, GRB.MINIMIZE) 
        # optimize SP
        self.SP.optimize()
                
    def set_SP(self):
        """ update must_include / cant_include constraints """
        for pair in self.SP_must_include:
            pi, pj = pair
            for i in range(self.graph.nodeNum):
                if i != pj and i!=0:
                    var_name = f"x[{pi},{i}]"
                    self.SP.getVarByName(var_name).setParam("UB", 0) #! check
                if i != pi and i!=0:
                    var_name = f"x[{i},{pj}]"
                    self.SP.getVarByName(var_name).setParam("UB", 0) #! check
        for pair in self.SP_cant_include:
            pi, pj = pair
            var_name = f"x[{pi},{pj}]" 
            self.SP.getVarByName(var_name).setParam("UB", 0) #! check
        self.SP.update()

    def reset_SP(self):
        """ reset must_include / cant_include constraints """
        for pair in self.SP_must_include:
            pi, pj = pair
            for j in range(self.graph.nodeNum):
                if j == pj:
                    continue
                var_name = f"x[{i},{j}]"
                self.SP.getVarByName(var_name).setParam("UB", 1) #! check
        for pair in self.SP_cant_include:
            pi, pj = pair
            var_name = f"x[{i},{j}]" 
            self.SP.getVarByName(var_name).setParam("UB", 1) #! check
        self.SP.update()
    
    def get_columns_from_SP_and_add(self):
        """ get columns from SP and add into RLMP """
        new_route = []
        route_length = 0
        col_num = self.graph.nodeNum
        new_column = np.zeros(self.graph.nodeNum)
        for SolutionNumber in range(self.SP.SolCount): #!check
            # get column from SP
            self.get_column_from_SP(new_route, route_length, new_column, SolutionNumber)
            self.add_column_into_RLMP(new_route, route_length, new_column)
        # display CG iteration info
        print(f"CG_iter {self.cg_iter_cnt}: RMPobj={self.RLMP.ObjVal}, SPobj={self.SP.ObjVal}")
    
    def get_column_from_SP(self, new_route, route_length, new_column, SolutionNumber=0):
        new_route.append(0)
        current_i = 0
        while True:
            for j in self.graph.feasibleNodeSet[current_i]:
                var_name = f"x[{current_i},{j}]"
                self.SP.setParam("SolutionNumber", SolutionNumber)
                var_val = round(self.SP.getVarByName(var_name).X)
                if var_val == 1:
                    route_length += self.graph.disMatrix[current_i, j]
                    new_route.append(j)
                    new_column[j] = 1
                    current_i = j
                    break
            if current_i == 0:
                break

    def add_column_into_RLMP(self, new_route, route_length, new_column):
        # update column pool
        new_column_name = "y[{}]".format(len(self.global_column_pool))
        self.global_column_pool[new_column_name] = new_route
        # update RLMP
        new_RLMP_column = gp.Column()
        new_RLMP_column.addTerms(new_column, self.RLMP.getConstrs())
        self.RLMP.addVar(obj=route_length, vtype=GRB.CONTINUOUS, column=new_RLMP_column, name=new_column_name)
    
    def solve_final_RMP_and_update_IPobj(self):
        # convert RLMP into RMP
        RMP = self.RLMP.copy()
        for var in RMP.getVars():
            var.vtype = 'B'
        # optimize RMP
        RMP.optimize()
        # update IP_obj if feasible
        if RMP.Status == 2:
            self.IP_obj = RMP.ObjVal
            self.way_of_opt = "By RMP"
        return RMP
    
    def get_columns_from_Labeling_and_add(self):
        # create tmp_graph, update must_include / cant_include constraints
        tmp_graph = deepcopy(self.graph)
        for pair in self.SP_must_include:
            pi, pj = pair
            for i in range(tmp_graph.nodeNum):
                if i!=pj and i!=0:
                    tmp_graph.disMatrix[pi][i] = np.inf
                if i!=pi and i!=0:
                    tmp_graph.disMatrix[i][pj] = np.inf
        for pair in self.SP_cant_include:
            pi, pj = pair
            tmp_graph.disMatrix[pi][pj] = np.inf
        tmp_graph.cal_feasibleNodeSet() # regenerate feasibleNodeSet
        # create labeling algorithm, set duals and solve
        alg = Labeling(tmp_graph, select_num=self.select_num)
        duals = np.zeros(tmp_graph.nodeNum)
        for di in range(tmp_graph.nodeNum):
            cons_name = f"R{di}"
            cons = self.RLMP.getConstrByName(cons_name)
            duals[di] = cons.Pi
        alg.set_dual(duals)
        routes, objs = alg.run()
        if len(objs) > 0:
            min_obj = min(objs)
        else:
            min_obj = np.inf
        # add routes into RLMP
        col_num = self.graph.nodeNum
        for route in routes: 
            # calculate route_length
            route_cost = self.cal_route_cost(route, tmp_graph)
            new_column = np.zeros(self.graph.nodeNum)
            for i in range(1, len(route)):
                new_column[route[i]] = 1
            self.add_column_into_RLMP(route, route_cost, new_column)
        # print(f"CG_iter {self.cg_iter_cnt}: RMPobj={self.RLMP.ObjVal}, min_obj={min_obj}, dominant_num={alg.total_dominant_num}")
        return min_obj
        
    def cal_route_cost(self, route, tmp_graph):
        weight_sum = 0
        volumn_sum = 0
        for i in route[1:-1]:
            weight_sum += tmp_graph.weight[i]
            volumn_sum += tmp_graph.volumn[i]
        chosen_w = None
        for w in range(tmp_graph.vehicleTypeNum):
            if weight_sum <= tmp_graph.weightCapacity[w] and volumn_sum <= tmp_graph.volumnCapacity[w]:
                chosen_w = w
                break
        assert chosen_w is not None, "ERROR: route overload"
        dist = 0
        cur_t = 0
        for i in range(1, len(route)):
            pi = route[i-1]
            pj = route[i]
            dist += tmp_graph.disMatrix[pi, pj]
            cur_t += tmp_graph.serviceTime[pi] + tmp_graph.disMatrix[pi, pj] / tmp_graph.speed
            assert cur_t <= tmp_graph.dueTime[pj], "ERROR: route overtime"
            cur_t = max(cur_t, tmp_graph.readyTime[pj])
        route_cost = tmp_graph.startPrice[chosen_w] + tmp_graph.meterPrice[chosen_w] * dist
        return route_cost
        
class BranchAndPrice():
    def __init__(self, graph, TimeLimit=1000):
        self.name = "BnP"
        # graph info
        self.graph = graph
        self.TimeLimit = TimeLimit
        self.global_column_pool = {}
        # build and set RLMP, SP
        self.RLMP = self.set_RLMP_model(graph)
        self.SP = self.set_SP_model(graph)    
        # create nodes
        self.root_node = BPNode(graph, self.RLMP, self.SP, self.global_column_pool)
        self.incumbent_node = self.root_node
        self.current_node = self.root_node
        # set strategies
        self.branch_strategy = "max_inf"
        self.search_strategy = "best_LB_first"
        # algorithm part
        self.node_list = []
        self.global_LB = np.inf
        self.global_UB = -np.inf
        # display parament
        self.iter_cnt = 0
        self.Gap = np.inf
        self.fea_sol_cnt = 0
        self.BP_tree_size = 0
        self.branch_var_name = ""

    def solution_init(self, graph):
        w = 0 # all smallest vehicle
        routes = []
        routes_cost = []
        for i in range(1, graph.nodeNum):
            route = [0] + [i] + [0]
            dist = graph.disMatrix[0, i] + graph.disMatrix[i, 0]
            route_cost = graph.startPrice[w] + graph.meterPrice[w] * dist
            routes.append(route)
            routes_cost.append(route_cost)
        routes_a = np.zeros((graph.nodeNum-1, graph.nodeNum))
        for i in range(graph.nodeNum-1):
            routes_a[i, 0] = 1
            routes_a[i, i+1] = 1
        return routes, routes_cost, routes_a

    def set_RLMP_model(self, graph):
        RLMP = gp.Model()
        # init solution
        routes, routes_cost, routes_a = self.solution_init(graph)
        for i in range(len(routes)):
            column_name = "y[{}]".format(i)
            self.global_column_pool[column_name] = routes[i]
        # add init solution in RLMP
        ## add variables
        y_list = list(range(len(routes)))
        y = RLMP.addVars(y_list, vtype="C", name="y")
        ## set objective
        RLMP.setObjective(gp.quicksum(y[i] * routes_cost[i] for i in range(len(routes))), GRB.MINIMIZE)
        ## set constraints
        RLMP.addConstrs(gp.quicksum(y[i] * routes_a[i, j] for i in range(len(routes))) >= 1 for j in range(graph.nodeNum))

        RLMP.setParam("OutputFlag", 0)
        RLMP.update()
        return RLMP
    
    def set_SP_model(self, graph):
        SP = gp.Model()
        ## add variables
        points = list(range(graph.nodeNum))
        A_list = [(i, j) for i in points for j in graph.feasibleNodeSet[i]]
        x = SP.addVars(A_list, vtype="B", name="x")
        t = SP.addVars(points, vtype="C", name="t")
        ## set objective 
        SP.setObjective(gp.quicksum(x[i, j] * graph.disMatrix[i][j] \
            for i in points for j in graph.feasibleNodeSet[i]))
        ## set constraints
        ### 1. flow balance
        SP.addConstrs(gp.quicksum(x[i, j] for j in graph.feasibleNodeSet[i] if j!=i)==1 for i in points[1:]) # depot not included
        SP.addConstrs(gp.quicksum(x[i, j] for i in graph.availableNodeSet[j] if i!=j)==1 for j in points[1:]) # depot not included
        ### 2. time window & sub-ring
        M = 1e7
        SP.addConstrs(t[i] + graph.disMatrix[i, j] + graph.serviceTime[i] - M * (1 - x[i, j]) <= t[j] for i, j in A_list if j!=0)
        SP.addConstrs(t[i] >= graph.readyTime[i] for i in points)
        SP.addConstrs(t[i] <= graph.dueTime[i] for i in points)

        # set model params
        SP.setParam("OutputFlag", 0)
        SP.update()
        return SP
    
    def root_init(self):
        self.root_node.solve_and_update()
        self.global_LB = self.root_node.LP_obj
        self.global_UB = self.root_node.IP_obj
        if (self.global_UB < np.inf):
            self.Gap = (self.global_UB - self.global_LB) / self.global_UB
        self.incumbent_node = self.root_node
        self.current_node = self.root_node
        if self.root_node.is_integer == False:
            self.branch(self.root_node)
    
    def search(self):
        """ best_LB_first: choose the node with best LB to search """
        best_node_i = 0
        if self.search_strategy == "best_LB_first":
            min_LB = np.inf
            for node_i in range(len(self.node_list)):
                LB = self.node_list[node_i].local_LB
                if LB < min_LB:
                    min_LB = LB
                    best_node_i = node_i
        best_node = self.node_list.pop(best_node_i)
        self.global_LB = min_LB # update global_LB
        return best_node
    
    def branch(self, node):
        # get flow of each arc
        flow_matrix = np.zeros((self.graph.nodeNum, self.graph.nodeNum))
        vars = node.RLMP.getVars()
        for var in vars:
            var_val = var.X
            var_name = var.VarName
            if var_val > 0:
                route = self.global_column_pool[var_name]
                for j in range(1, len(route)):
                    pi = route[j-1]
                    pj = route[j]
                    flow_matrix[pi][pj] += var_val
        # max_inf: choose the arc farthest to integer to branch 
        best_i = best_j = 0
        if self.branch_strategy == "max_inf":
            max_inf = -np.inf
            for pi in range(self.graph.nodeNum):
                for pj in range(self.graph.nodeNum):
                    if flow_matrix[pi, pj] > 1:
                        continue
                    cur_inf = abs(flow_matrix[pi, pj] - round(flow_matrix[pi, pj]))
                    if cur_inf > max_inf:
                        max_inf = cur_inf
                        best_i = pi
                        best_j = pj
        # branch on the chosen variable
        self.branch_var_name = f"x[{best_i},{best_j}]"

        ## 1. branch left, must include xij
        leftNode = node.generate()
        ### delete routes constains i or j, but xij != 1
        for var in vars:
            var_name = var.VarName
            route = self.global_column_pool[var_name]
            if best_i not in route and best_j not in route:
                continue
            elif best_i in route and best_j in route:
                skip_flag = False
                for i in range(len(route)-1):
                    if route[i] == best_i and route[i+1] == best_j:
                        skip_flag = True
                        break
                if skip_flag:
                    continue
            elif (best_i == 0) and (best_i in route and best_j not in route):
                continue
            elif (best_j == 0) and (best_j in route and best_i not in route):
                continue
            var_name = var.VarName
            left_var = leftNode.RLMP.getVarByName(var_name)
            left_var.ub = 0 # make var_i invalid
        ### add into must include list
        leftNode.SP_must_include.append([best_i, best_j])
        self.node_list.append(leftNode)
        self.BP_tree_size+=1

        ## 2. branch right, cant include xij
        rightNode = node.generate()
        ### delete routes with xij == 1
        for var in vars:
            var_name = var.VarName
            route = self.global_column_pool[var_name]
            for i in range(len(route)-1):
                if route[i] == best_i and route[i+1] == best_j:
                    var_name = var.VarName
                    right_var = rightNode.RLMP.getVarByName(var_name)
                    right_var.ub = 0 # make var_i invalid
        ### add into cant include list
        rightNode.SP_cant_include.append([best_i, best_j])
        self.node_list.append(rightNode)
        self.BP_tree_size+=1

    def display_MIP_logging(self):
        """
        Show the MIP logging.

        :param iter_cnt:
        :return:
        """
        self.end_time = time.time()
        if (self.iter_cnt <= 0):
            print('|%6s  |' % 'Iter', end='')
            print(' \t\t %1s \t\t  |' % 'BB tree', end='')
            print('\t %10s \t |' % 'Current Node', end='')
            print('    %11s    |' % 'Best Bounds', end='')
            print(' %8s |' % 'incumbent', end='')
            print(' %5s  |' % 'Gap', end='')
            print(' %5s  |' % 'Time', end='')
            print(' %6s |' % 'Feasible', end='')
            print('     %10s      |' % 'Branch Var', end='')
            print()
            print('| %4s   |' % 'Cnt', end='')
            print(' %5s |' % 'Depth', end='')
            print(' %8s |' % 'ExplNode', end='')
            print(' %10s |' % 'UnexplNode', end='')
            print(' %4s |' % 'InfCnt', end='')
            print('    %3s   |' % 'Obj', end='')
            print('%10s |' % 'PruneInfo', end='')
            print(' %7s |' % 'Best UB', end='')
            print(' %7s |' % 'Best LB', end='')
            print(' %8s |' % 'Objective', end='')
            print(' %5s  |' % '(%)', end='')
            print(' %5s  |' % '(s)', end='')
            print(' %8s |' % ' Sol Cnt', end='')
            print(' %7s  |' % 'Max Inf', end='')
            print(' %7s  |' % 'Max Inf', end='')
            print()
        if(self.incumbent_node == None):
            print('%2s' % ' ', end='')
        elif(self.incumbent_node.way_of_opt == 'By Rouding' and self.incumbent_node.has_showed_way_of_opt == False):
            print('%2s' % 'R ', end='')
            self.incumbent_node.has_showed_way_of_opt = True
        elif (self.incumbent_node.way_of_opt == 'By Simplex' and self.incumbent_node.has_showed_way_of_opt == False):
            print('%2s' % '* ', end='')
            self.incumbent_node.has_showed_way_of_opt = True
        # elif (self.current_node.has_a_int_sol_by_heur == True and incumbent_node.has_showed_heu_int_fea == False):
        #     print('%2s' % 'H ', end='')
        #     self.incumbent_node.has_showed_heu_int_fea = True
        else:
            print('%2s' % ' ', end='')

        print('%3s' % self.iter_cnt, end='')
        print('%10s' % self.current_node.depth, end='')
        print('%9s' % self.iter_cnt, end='')
        print('%12s' % len(self.node_list), end='')
        if (len(self.current_node.inf_var_list) > 0):
            print('%11s' % len(self.current_node.inf_var_list), end='')
        else:
            if (self.current_node.RLMP.status == 2):
                print('%11s' % 'Fea Int',
                        end='')  # indicates that this is a integer feasible solution, no variable is infeasible
            elif (self.current_node.RLMP.status == 3):
                print('%11s' % 'Inf Model',
                        end='')  # indicates that this is a integer feasible solution, no variable is infeasible
            else:
                print('%11s' % '---',
                        end='')  # indicates that this is a integer feasible solution, no variable is infeasible
        if (self.current_node.RLMP.status == 2):
            print('%12s' % round(self.current_node.RLMP.ObjVal, 2), end='')
        else:
            print('%12s' % '---', end='')
        print('%10s' % self.current_node.prune_info, end='')
        print('%12s' % round(self.global_UB, 2), end='')
        print('%10s' % round(self.global_LB, 2), end='')
        if(self.incumbent_node == None):
            print('%11s' % '---', end='')
        else:
            print('%11s' % round(self.incumbent_node.IP_obj, 2), end='')
        if (self.Gap != '---'):
            print('%9s' % round(100 * self.Gap, 2), end='%')
        else:
            print('%8s' % 100 * self.Gap, end='')
        print('%8s' % round(self.end_time - self.start_time, 0), end='s')
        print('%9s' % self.fea_sol_cnt, end=' ')
        print('%14s' % self.branch_strategy, end='')
        print('%9s' % self.branch_var_name, end='')
        print()

    def display_result(self):
        self.CPU_time = time.time() - self.start_time
        print('\n')
        if len(self.node_list) == 0:
            print("Unexplored node list empty")
        else:
            print("Global LB and UB meet")
        print("Branch and bound terminates !!!")
        print("\n\n ------------ Summary ------------")
        print("Incumbent Obj: {}".format(self.incumbent_node.IP_obj))
        print("Gap: {}%".format(round(self.Gap * 100) if self.Gap < np.inf else 'inf')  )
        print("BB tree size: {}".format(self.BP_tree_size))
        print("CPU time: {}s".format(self.CPU_time))
        print(" --------- Solution  --------- ")
        for key in self.incumbent_node.x_int_sol.keys():
            if self.incumbent_node.x_int_sol[key] == 1:
                print('{} = {}: {}'.format(key, self.incumbent_node.x_int_sol[key], self.global_column_pool[key]))

    def get_routes(self, node):
        RMP = node.solve_final_RMP_and_update_IPobj()
        routes = []
        for var in RMP.getVars():
            var_name = var.VarName
            if var.X > 0.5:
                route = self.global_column_pool[var_name] 
                routes.append(route)
        return routes

    def show_result(self, plot=True):
        # display result
        routes = self.get_routes(self.incumbent_node)
        # print("Incumbent IP_Obj = {}".format(self.incumbent_node.IP_obj))
        print("Incumbent IP_Obj = {}".format(self.graph.cal_objective(routes)))
        print("Routes:")
        for ri in range(len(routes)):
            print("{}: {}".format(ri, routes[ri]))
        self.graph.render(routes)

    def run(self):
        self.start_time = time.time()
        """ initalize the node """
        self.root_init() # solve root node and update global_LB/UB and branch
        """ branch and bound """
        while (time.time()-self.start_time) < self.TimeLimit and len(self.node_list) > 0 and self.global_LB < self.global_UB:
            """ search part """
            self.current_node = self.search() # get a node from node_list

            """ solve and prune """
            incum_update = False # record whether incumbent updated
            # prune1: By Bnd
            if self.current_node.local_LB >= self.global_UB: 
                self.current_node.prune_info = 'By Bnd'
            else:
                # prune2: By Inf
                self.current_node.solve_and_update()
                if not self.current_node.is_feasible:            
                    self.current_node.prune_info = 'By Inf'
                else:
                    # prune3: By Opt
                    if self.current_node.is_integer:                 
                        self.fea_sol_cnt += 1
                        if self.current_node.IP_obj < self.global_UB: # update best solution
                            self.global_UB = self.current_node.IP_obj
                            self.incumbent_node = self.current_node
                            incum_update = True
                            if (self.global_UB < np.inf):
                                self.Gap = (self.global_UB - self.global_LB) / self.global_UB
                        self.current_node.prune_info = 'By Opt'

            """ branch part """
            if self.current_node.prune_info == '---': # if not been pruned
                self.branch(self.current_node)
        
            """ display logging """
            # if self.iter_cnt % 100 == 0 or incum_update: # display when iter 10 times or when update incumbent
            self.display_MIP_logging()
                
            self.iter_cnt += 1

        if len(self.node_list) == 0:
            self.global_LB = self.global_UB
        self.Gap = (self.global_UB - self.global_LB) / self.global_UB
        self.display_MIP_logging()
        """ display result """
        self.display_result()
        return self.get_routes(self.incumbent_node)

if __name__ == "__main__":
    graph = GraphTools.Graph()    
    alg = BranchAndPrice(graph)
    alg.run() # run B&P
    alg.show_result()
