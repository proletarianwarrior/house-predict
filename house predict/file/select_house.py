# -*- coding: utf-8 -*-
# @Time : 2022/9/19 13:48
# @Author : DanYang
# @File : select_house.py
# @Software : PyCharm
import latitude_longitude
import pymongo

client = pymongo.MongoClient(host='localhost', port=27017)
db = client['houses']


if __name__ == '__main__':
    selection = latitude_longitude.LatitudeLongitude()

    hospital = selection.get_poi_loc('三级甲等医院')
    school = selection.get_poi_loc('中学')
    p_school = selection.get_poi_loc('小学')
    school.update(p_school)
    subway = selection.get_poi_loc('地铁')
    commerce = selection.get_poi_loc('购物中心;商场')

    hospital_collection = db['hospital']
    school_collection = db['school']
    subway_collection = db['subway']
    commerce_collection = db['commerce']

    hospital_collection.insert_one(hospital)
    school_collection.insert_one(school)
    subway_collection.insert_one(subway)
    commerce_collection.insert_one(commerce)
