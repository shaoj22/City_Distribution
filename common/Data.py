# Data class, containing all data of instance
# author: Charles Lee
# date: 2022.12.10

import pandas as pd
import numpy as np

import sys
sys.path.append('common')
from Map import Map
from Node import Node
from Order import Order
from VehicleType import VehicleType

class Data():
    def __init__(self):
        map, order_list, vehicle_type_list = read_data()
        self.map = map
        self.vehicle_type_list = vehicle_type_list
        self.order_list = order_list
    
def read_data():
    """ 
    read data from files 

    standard units:
        distance: meter
        time: second
        weight: kg
        volumn: m3
    """
    map_df = pd.read_excel("距离矩阵.xlsx")
    order_df = pd.read_excel("订单.xlsx")
    vehicle_df = pd.read_excel("车型.xlsx")

    # 1. get map
    node_list = [] # get node list without repitition
    depot_address = "北京市通州区通州区招商局物流集团(北京有限公司)"
    depot_posXY_str = "116.577720,39.759990"
    posX_str, posY_str = depot_posXY_str.split(",")
    posX = eval(posX_str)
    posY = eval(posY_str)
    node = Node(posX, posY, depot_address)
    node_list.append(node)
    tmp_tabu = {depot_address: 1}
    for i in range(len(order_df)):
        address = order_df["送货地址"][i]
        if address in tmp_tabu:
            continue
        else:
            tmp_tabu[address] = 1
        posXY_str = order_df["经纬度"][i]
        posX_str, posY_str = posXY_str.split(",")
        posX = eval(posX_str)
        posY = eval(posY_str)
        node = Node(posX, posY, address)
        node_list.append(node)
    disMatrix = np.zeros((len(node_list), len(node_list)))
    pos_list = list(map_df.columns)
    for i in range(len(node_list)):
        for j in range(len(node_list)):
            posi = "{:.6f},{:.6f}".format(node_list[i].posX, node_list[i].posY)
            posj = "{:.6f},{:.6f}".format(node_list[j].posX, node_list[j].posY)
            pi = pos_list.index(posi)
            pj = pos_list.index(posj)
            disMatrix[i, j] = map_df.iloc[pi-1, pj]
    map = Map(node_list, disMatrix)

    # 2. get order_list
    order_list = []
    for i in range(len(order_df)):
        date = order_df["送货日期"][i]
        quantity = float(order_df["数量"][i])
        quality = float(order_df["吨位"][i]) * 1000
        volumn = float(order_df["体积(m3)"][i])
        place_str = order_df["送货地址"][i]
        place = map.get_index_of_place(place_str)
        readyTime = transfer_time(order_df["最早送到货的时间"][i])
        dueTime = transfer_time(order_df["最晚送到货的时间"][i])
        if dueTime == 0:
            dueTime = 24 * 3600 # 24:00
        packSpeed = float(order_df["送货点卸货效率(kg/h)"][i]) / 3600
        waitTime = float(order_df["送货点平均等待时长(h)"][i]) * 3600
        order = Order(date, quantity, quality, volumn, place, readyTime, dueTime, packSpeed, waitTime)
        order_list.append(order)
    
    # 3. get vehicle_type_list
    vehicle_type_list = []
    for i in range(len(vehicle_df)):
        typeName = vehicle_df["车型"][i]
        weight_capacity = int(vehicle_df["车辆最大装载（吨）"][i]) * 1000
        volumn_capacity = int(vehicle_df["车辆最大装载（立方）"][i])
        startPrice = int(vehicle_df["起步价（包含一个送货点）"][i])
        pointPrice = int(vehicle_df["点位费（从第二个送货点开始，元/点）"][i])
        meterPrice = float(vehicle_df["每公里油钱（元/km）"][i]) / 1000 # yuan/meter
        speed = 35 * 1000 / 3600 # 35km/h
        vehicle_type = VehicleType(typeName, weight_capacity, volumn_capacity, startPrice, pointPrice, meterPrice, speed) 
        vehicle_type_list.append(vehicle_type)
    
    return map, order_list, vehicle_type_list

def transfer_time(datetime):
    """ transfer time string to int time (second)

    Args:
        datetime (datetime): time(hour, second)
    Returns:
        time (int): seconds
    """
    hour = datetime.hour
    minute = datetime.minute
    second = datetime.second
    time = hour * 3600 + minute * 60 + second
    return time

if __name__ == "__main__":
    data = Data() 
    data.map.render()