# -*- coding: utf-8 -*-
# @Time : 2022/9/21 19:46
# @Author : DanYang
# @File : minmax.py
# @Software : PyCharm
import numpy as np
import json


def min_max(data_list):
    mean = np.mean(data_list)
    std = np.std(data_list)
    return (data_list-mean)/std


def split_data():
    with open('new_houses.json', 'r', encoding='gbk') as file:
        results = json.loads(file.read())
    use = [i['用途'] for i in results]
    heat = [i['供暖'] for i in results]
    floor = [i['楼层'] for i in results]
    year = [i['年份'] for i in results]
    room = [i['户型(/室)'] for i in results]
    square = [i['面积(/平方米)'] for i in results]
    towards = [i['朝向'] for i in results]
    money = [i['均价(元/平方米)'] for i in results]
    loa = [i['经纬度'] for i in results]

    result_list = [use, heat, floor, year, room, square, towards, money]
    return [min_max(np.array(i)) for i in result_list] + loa


