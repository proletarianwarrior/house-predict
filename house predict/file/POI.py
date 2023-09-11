# -*- coding: utf-8 -*-
# @Time : 2022/9/21 20:29
# @Author : DanYang
# @File : POI.py
# @Software : PyCharm
import minmax
import json
import common


def merge(name):
    with open(name, 'r', encoding='utf-8') as file:
        return list(json.loads(file.read())[0].values())


def main():
    result = []
    names = ['./data/hospital.json', './data/school.json',
             './data/subway.json', './data/commerce.json']
    for name in names:
        result.extend(merge(name))

    return result


def poi():
    results = minmax.split_data()
    dis_list = main()
    nums = []
    for result in results[-1]:
        number = 0
        for dis in dis_list:
            if common.distance(result, dis) <= 3000:
                number += 1
        nums.append(number)
    return results + [nums]

