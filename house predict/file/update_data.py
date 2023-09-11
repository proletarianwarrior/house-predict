# -*- coding: utf-8 -*-
# @Time : 2022/9/21 17:15
# @Author : DanYang
# @File : update_data.py
# @Software : PyCharm
import json


def pre_data(data):
    if data['用途'] == '普通住宅':
        data['用途'] = 1
    else:
        data['用途'] = 2

    if data['供暖'] == '集体供暖':
        data['供暖'] = 2
    else:
        data['供暖'] = 1

    if data['楼层'] == '中':
        data['楼层'] = 2
    else:
        data['楼层'] = 1

    data['年份'] = 2022 - int(data['年份'])
    data['户型(/室)'] = int(data['户型(/室)'])
    data['面积(/平方米)'] = float(data['面积(/平方米)'])
    if '南' in data['朝向']:
        data['朝向'] = 3
    elif '北' in data['朝向']:
        data['朝向'] = 2
    else:
        data['朝向'] = 1

    data['均价(元/平方米)'] = float(data['均价(元/平方米)'])
    return data


if __name__ == '__main__':
    with open('houses.json', 'r', encoding='utf-8') as file:
        results = json.loads(file.read())
    for i, result in enumerate(results):
        print(i, result)
        new_result = pre_data(result)
        results[i] = new_result

    with open('new_houses.json', 'w') as file:
        file.write(json.dumps(results, indent=3, ensure_ascii=False))



