# Graph class, containing all data of instance preprocessed
# author: Charles Lee
# date: 2022.12.15

import sys
sys.path.append('common')
import Data as DataTools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import seaborn as sns
import pandas as pd

class Graph():
    def __init__(self, limit_nodeNum=None, limit_vehicleTypeNum=None):
        data = DataTools.Data()
        self._limit_nodeNum = limit_nodeNum
        self._limit_vehicleTypeNum = limit_vehicleTypeNum
        self._preprocess(data)
        # evaluate paraments
        self.punish_for_overload = 1e5
        self.punish_for_overtime = 1e3
    
    def _preprocess(self, data):
        """
        Order Batching and Preprocess Data
        
        Data:
            nodeNum (int == N)
            disMatrix (ndarray NxN)
            location (ndarray Nx2)
            weight (ndarray N)
            volumn (ndarray N)
            readyTime (ndarray N)
            dueTime (ndarray N)
            serviceTime (ndarray N)
            feasibleNodeSet (List[List[int]] Nx?)
            availableNodeSet (List[List[int]] Nx?)

            vehicleTypeNum (int == W)
            vehicleTypeName (List[string] W)
            weightCapacity (ndarray W)
            volumnCapacity (ndarray W)
            startPrice (ndarray W)
            meterPrice (ndarray W)
            speed (float)

        """
        # get data from data object
        if self._limit_vehicleTypeNum is None:
            vehicle_type_list = data.vehicle_type_list
        else:
            vehicle_type_list = data.vehicle_type_list[-self._limit_vehicleTypeNum:]
        # preprocess data
        ## 1. order dispatching
        if self._limit_nodeNum is None:
            order_list = self._order_batching(data.order_list, vehicle_type_list)
        else:
            order_list = self._order_batching(data.order_list, vehicle_type_list)[:self._limit_nodeNum]
        ## 2. combine data.order_list and map data to self.node_list
        node_list = [] # node_list here different from node_list in data.map
        depot_node = data.map.node_list[0].copy()
        depot_node.set_order_information(DataTools.Order())
        node_list.append(depot_node)
        for order in order_list:
            ni = order.place
            node = data.map.node_list[ni].copy()
            node.set_order_information(order)
            node_list.append(node)
        # 3. calculate other data needed
        self.nodeNum = len(node_list)
        self.vehicleTypeNum = len(vehicle_type_list)
        self.disMatrix = np.zeros((self.nodeNum, self.nodeNum))
        for i in range(self.nodeNum):
            for j in range(self.nodeNum):
                ni = node_list[i].place
                nj = node_list[j].place
                self.disMatrix[i, j] = data.map.disMatrix[ni, nj]
        self.speed = vehicle_type_list[0].speed
        # 4. preprocess node_list and vehicle_type_list
        self.location = np.zeros((self.nodeNum, 2))
        self.weight = np.zeros(self.nodeNum)
        self.volumn = np.zeros(self.nodeNum)
        self.readyTime = np.zeros(self.nodeNum)
        self.dueTime = np.zeros(self.nodeNum)
        self.serviceTime = np.zeros(self.nodeNum)
        for i in range(self.nodeNum):
            self.location[i] = [node_list[i].posX, node_list[i].posY]
            self.weight[i] = node_list[i].quality
            self.volumn[i] = node_list[i].volumn
            self.readyTime[i] = node_list[i].readyTime
            self.dueTime[i] = node_list[i].dueTime
            self.serviceTime[i] = node_list[i].serviceTime
        self.vehicleTypeName = ['' for _ in range(self.vehicleTypeNum)]
        self.weightCapacity = np.zeros(self.vehicleTypeNum)
        self.volumnCapacity = np.zeros(self.vehicleTypeNum)
        self.startPrice = np.zeros(self.vehicleTypeNum)
        self.meterPrice = np.zeros(self.vehicleTypeNum)
        for w in range(self.vehicleTypeNum):
            self.vehicleTypeName[w] = vehicle_type_list[w].typeName
            self.weightCapacity[w] = vehicle_type_list[w].weight_capacity
            self.volumnCapacity[w] = vehicle_type_list[w].volumn_capacity
            self.startPrice[w] = vehicle_type_list[w].startPrice
            self.meterPrice[w] = vehicle_type_list[w].meterPrice
        # 5. calculate feasibleNodeSet
        self.cal_feasibleNodeSet()
    
    def _order_batching(self, order_list, vehicle_type_list):
        weight_capacity = vehicle_type_list[0].weight_capacity
        volumn_capacity = vehicle_type_list[0].volumn_capacity
        new_order_list = []
        check_order = order_list[0]
        for ri in range(1, len(order_list)):
            if order_list[ri].place == check_order.place and \
               order_list[ri].quality + check_order.quality <= weight_capacity and \
               order_list[ri].volumn + check_order.volumn <= volumn_capacity:
                # combine the order if satisfy conditions
                check_order.quantity += order_list[ri].quantity
                check_order.quality += order_list[ri].quality
                check_order.volumn += order_list[ri].volumn
            else:
                # add to list if not satisfy 
                new_order_list.append(check_order)
                check_order = order_list[ri]
        else:
            new_order_list.append(check_order)
        return new_order_list
                
    def cal_feasibleNodeSet(self):
        """
        update feasible node set
        """
        self.infeasibleNodeSet = [[] for _ in range(self.nodeNum)]
        self.feasibleNodeSet = [[] for _ in range(self.nodeNum)]
        self.availableNodeSet = [[] for _ in range(self.nodeNum)]
        for i in range(self.nodeNum):
            for j in range(self.nodeNum):
                if i == j:
                    continue
                if self.readyTime[i] + self.serviceTime[i] + self.disMatrix[i, j] / self.speed <= self.dueTime[j] and i!=j \
                   and self.weight[i] + self.weight[j] <= max(self.weightCapacity) \
                   and self.volumn[i] + self.volumn[j] <= max(self.volumnCapacity):
                    self.feasibleNodeSet[i].append(j)
                    self.availableNodeSet[j].append(i)
                else:
                    self.infeasibleNodeSet[i].append(j)

    def cal_objective(self, routes):
        """ calculate fitness(-obj) 
        obj = distance_cost + overload_cost + overtime_cost
        """
        route_obj_list = []
        for route in routes:
            route_obj = self.cal_vehicle_objective(route)
            route_obj_list.append(route_obj)
        obj = sum(route_obj_list)
        return obj

    def cal_sperate_objective(self, routes):
        """
        calculate startCost and meterCost seperately
        (!!Assume no constraints violated!!)
        """
        startCost = meterCost = 0
        for route in routes:
            vi = self.choose_vehicle(route)
            startCost += self.startPrice[vi]
            total_dist = 0
            for i in range(len(route)-1):
                total_dist += self.disMatrix[route[i], route[i+1]]
            meterCost += self.meterPrice[vi] * total_dist
        return startCost, meterCost

    def cal_vehicle_objective(self, route):
        if len(route) <= 2: # empty vehicle
            return 0
        route_obj = 0
        # 1. check overload and choose vehicle
        vi = self.choose_vehicle(route) 
        if vi is None: # overload
            return self.punish_for_overload
        # 2. calculate obj (cost)
        ## distance cost
        route_obj += self.cal_distance_cost(route, vi)
        ## time window cost
        route_obj += self.cal_time_window_cost(route)
        return route_obj

    def choose_vehicle(self, route):
        # choose vehicle according to weight/volumn load
        weight_sum = 0
        volumn_sum = 0
        for pi in route[1:-1]:
            weight_sum += self.weight[pi]
            volumn_sum += self.volumn[pi]
        for vi in range(self.vehicleTypeNum): # vehicle_type rank from small to big
            if self.weightCapacity[vi] >= weight_sum and \
               self.volumnCapacity[vi] >= volumn_sum:
                return vi
        return None # None means overload
    
    def cal_distance_cost(self, route, vi):
        distance_cost = 0
        # add start Price of each vehicle
        distance_cost += self.startPrice[vi]
        # calculate total distance and add distance price
        total_dist = 0
        for i in range(len(route)-1):
            total_dist += self.disMatrix[route[i], route[i+1]]
        distance_cost += self.meterPrice[vi] * total_dist
        return distance_cost
    
    def cal_time_window_cost(self, route):
        # if violate time window, give punishment
        tw_cost = 0
        cur_t = 0
        for i in range(len(route)-2):
            pi = route[i]
            pj = route[i+1]
            dist = self.disMatrix[pi, pj]
            cur_t += dist / self.speed + self.serviceTime[pi]
            cur_t = max(cur_t, self.readyTime[pj])
            if cur_t > self.dueTime[pj]:
                tw_cost += self.punish_for_overtime 
        return tw_cost

    def render(self, routes):
        # plot routes 
        X = self.location[:, 0]
        print(self.location)
        Y = self.location[:, 1]
        plt.scatter(X, Y, c='black')
        for route in routes:
            X = self.location[route, 0]
            Y = self.location[route, 1]
            plt.plot(X, Y)
        plt.show()



    def visualize_route(self, coordinates, paths):
        sns.set(style="whitegrid")  # 设置Seaborn样式
        plt.figure(figsize=(12, 10))  # 设置图表大小
        # 不同路径使用不同颜色和线条样式
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        line_styles = ['-', '--', '-.', ':']
        for idx, path in enumerate(paths):
            path_coordinates = [coordinates[idx] for idx in path]
            path_x, path_y = zip(*path_coordinates)
            color = colors[idx % len(colors)]
            line_style = line_styles[idx % len(line_styles)]
            plt.plot(path_x, path_y, marker='o', linewidth=2, color=color, linestyle=line_style, label=f'Path {idx+1}')
        warehouse_x, warehouse_y = coordinates[0]
        plt.plot(warehouse_x, warehouse_y, marker='s', color='red', markersize=10, label='Warehouse')
        plt.title('Route Planning Visualization', fontsize=18)  # 增加标题字体大小
        plt.xlabel('Longitude', fontsize=16)  # 坐标轴标签字体大小
        plt.ylabel('Latitude', fontsize=16)  # 坐标轴标签字体大小
        plt.legend(fontsize=14)  # 图例字体大小
        plt.grid(True)
        plt.show()







    def output_batch_orders(self):
        """
        output excel file "分批订单" of batched orders
        """
        order_df = pd.DataFrame()
        order_idx_list = list(range(1, self.nodeNum))
        order_df["orderID"] = order_idx_list
        order_df["weight(kg)"] = self.weight[order_idx_list]
        order_df["volumn(m3)"] = self.volumn[order_idx_list]
        order_df["readyTime(s)"] = self.readyTime[order_idx_list]
        order_df["dueTime(s)"] = self.dueTime[order_idx_list]
        order_df["serviceTime(s)"] = self.serviceTime[order_idx_list]
        order_df["location"] = ["{}, {}".format(location[0], location[1]) for location in self.location[1:]]
        order_df.to_excel("分批订单.xlsx", index=False)
    


if __name__ == "__main__":
    graph = Graph()







