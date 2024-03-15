# auto experiments
# author: Charles Lee
# date: 2021.01.05

import common.Graph as GraphTools
from VNS import *
from ALNS import *
from Gurobi import *
from BranchAndPrice import *
from ColumGeneration import *
import time

def generate_instances():
    intances = []
    for limit_nodeNum in [55]:
        for limit_vehicleTypeNum in [7]:
            intances.append(GraphTools.Graph(limit_nodeNum, limit_vehicleTypeNum))
    return intances

def run_algorithm(alg, graph):
    # record results in list
    result = []
    # eval algorithm
    alg = alg(graph)
    result.append(alg.name) # alg_name

    # run algorithm
    start = time.time()
    routes = alg.run()
    end = time.time()

    # evaluate
    obj = graph.cal_objective(routes)
    startCost, meterCost = graph.cal_sperate_objective(routes)
    result.append(obj) # cost
    result.append(startCost) # start_cost
    result.append(meterCost) # meter_cost
    result.append(end - start) # time_consume
    return result

if __name__ == "__main__":
    graph_list = generate_instances()
    alg_list = [Solomon_Insertion, VNS, ALNS, BranchAndPrice, GurobiSolver1, ColumnGeneration]
    result_list = []
    for alg in alg_list[-2:]:
        for graph in graph_list:
            result = run_algorithm(alg, graph)
            result = [graph._limit_nodeNum, graph._limit_vehicleTypeNum] + result # node_num, vehicle_type_num
            result_list.append(result)
            # save file
            result_df = pd.DataFrame(result_list)
            result_df.columns = ["node_num", "vehicle_type_num", "alg_name", "total_cost", "start_cost", "meter_cost", "time_consume"]
            result_df.to_excel("experiment_result.xlsx")
    print("finished")

