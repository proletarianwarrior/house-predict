# -*- coding: utf-8 -*-
# @Time : 2022/9/18 17:26
# @Author : DanYang
# @File : latitude_longitude.py
# @Software : PyCharm
import requests
from bs4 import BeautifulSoup


class LatitudeLongitude:
    def __init__(self):
        self.SESSION = requests.Session()
        self.BASE_URL = 'https://restapi.amap.com/v3/geocode/geo?' \
                        'address={address}&output=XML&city=西安&key=dd36390ef3c7b9ce04136c408a65e854'
        self.POI_URL = 'https://restapi.amap.com/v3/place/text?types={types}' \
                       '&city=610103&offset=20&page={page}&key=dd36390ef3c7b9ce04136c408a65e854'

    def submit_house(self, params, url):
        new_url = url.format(**params)
        response = self.SESSION.get(new_url)
        if url == self.BASE_URL:
            return response.text
        else:
            return response.json()

    def get_loa(self, name):
        params = {'address': name}
        html = self.submit_house(params, self.BASE_URL)
        try:
            content = BeautifulSoup(html)
            location = content.location.string
        except AttributeError:
            return None
        location = [float(i) for i in location.split(',')]
        return location

    def get_poi_loc(self, types):
        params = {
            'types': types,
            'page': '1'
        }
        html = self.submit_house(params, self.POI_URL)
        count = html['count']
        total_page = int(count) // 20 + 1
        names = []
        locations = []
        for i in range(1, total_page + 1):
            params['page'] = str(i)
            html = self.submit_house(params, self.POI_URL)
            for poi in html['pois']:
                names.append(poi['name'])
                locations.append([float(i) for i in poi['location'].split(',')])
        return dict(zip(names, locations))