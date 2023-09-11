# -*- coding: utf-8 -*-
# @Time : 2022/9/17 0:09
# @Author : DanYang
# @File : crawler.py
# @Software : PyCharm
import csv
import aiohttp
import asyncio
import logging
import aiofiles
import json
import re
from motor.motor_asyncio import AsyncIOMotorClient
from pyquery import PyQuery as pq

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')


class Crawler:
    def __init__(self):
        self.BASE_URL = 'https://xa.ke.com/ershoufang/beilin/{page}{heating}{application}/'

        self.CONCURRENCY = 10
        self.SEMAPHORE = asyncio.Semaphore(self.CONCURRENCY)
        self.TIMEOUT = aiohttp.ClientTimeout(total=10)

        self.MONGO_CONNECTION_STRING = 'mongodb://localhost:27017'
        self.MONGO_DB_NAME = 'houses'
        self.MONGO_COLLECTION_NAME = 'houses'

    async def scrape_html(self, url):
        session = aiohttp.ClientSession(timeout=self.TIMEOUT)
        logging.info('scraping %s', url)
        async with self.SEMAPHORE:
            try:
                response = await session.get(url)
                logging.info('success scrap %s', url)
                return (await response.text()).replace('\xa9', '')
            except aiohttp.ClientError:
                logging.error('error occurred while scraping %s', url, exc_info=True)
                return None
            finally:
                await session.close()

    async def get_area(self):
        async with aiofiles.open('area.json', 'r') as file:
            return json.loads(await file.read())

    async def get_url(self, params):
        return self.BASE_URL.format(**params)

    async def get_page(self, html):
        content = pq(html)
        div = content('div[comp-module="page"]')
        page = re.search('"totalPage":([0-9]*)', div.attr('page-data')).group(1)
        return int(page)

    async def parse_html(self, html):
        content = pq(html)
        try:
            address_list = content('div[data-component="list"] ul.sellListContent li div div.address').items()
            flood = [address('div.flood div a').text() for address in address_list]
            patterns = re.compile('<span class="houseIcon"></span>(.*?)</div>', re.S)
            house_info = [re.sub('\s', '', house) for house in re.findall(patterns, html)]
            price_info_list = content('div[data-component="list"] ul.sellListContent li div div.priceInfo').items()
            price = [price_info('div.unitPrice span').text() for price_info in price_info_list]
            return flood, house_info, price
        except AttributeError:
            logging.error('error occurred while parsing %s...', html[:50])
            return None

    async def split_text(self, tuple_text):
        flood, house_info, price = tuple_text
        if not (len(flood) == len(house_info) == len(price)):
            return None
        house_info = [house.split('|') for house in house_info]

        flood_result = []
        price_result = []
        house_result = []
        for i, house in enumerate(house_info):
            if len(house) != 5:
                continue
            try:
                house[0] = re.search('([低中高地下室]*)', house[0]).group(1)
                house[1] = re.search('([0-9]*)年', house[1]).group(1)
                house[2] = re.search('([0-9]*)室', house[2]).group(1)
                house[3] = re.search('([0-9.]*)平米', house[3]).group(1)
                price[i] = re.search('([0-9]*)元', price[i].replace(',', '')).group(1)

                flood_result.append(flood[i])
                price_result.append(price[i])
                house_result.append(house)
            except AttributeError:
                continue

        return flood_result, house_result, price_result

    async def save_data_mongo_db(self, data_tuple):
        logging.info('saving data')

        client = AsyncIOMotorClient(self.MONGO_CONNECTION_STRING)
        db = client[self.MONGO_DB_NAME]
        collection = db[self.MONGO_COLLECTION_NAME]

        floods, houses, prices, application, heat = data_tuple
        fieldnames = ['小区名称', '用途', '供暖', '楼层', '年份', '户型(/室)', '面积(/平方米)', '朝向', '均价(元/平方米)']
        for flood, house, price in zip(floods, houses, prices):
            results = [flood, application, heat] + house + [price]
            condition = dict(zip(fieldnames, results))
            await collection.insert_one(condition)

        logging.info('success save data')

    async def save_data_csv(self, data_tuple):
        logging.info('saving data')
        floods, houses, prices, application, heat = data_tuple
        fieldnames = ['小区名称', '用途', '供暖', '楼层', '年份', '户型(/室)', '面积(/平方米)', '朝向', '均价(元/平方米)']
        for flood, house, price in zip(floods, houses, prices):
            results = [flood, application, heat] + house + [price]
            condition = dict(zip(fieldnames, results))
            with open('data.csv', 'a', newline='', encoding='gbk') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                try:
                    writer.writerow(condition)
                except AttributeError:
                    pass

    async def main(self):
        area = (await self.get_area())[0]
        applications = area['application']
        heats = area['heating']
        for application in applications:
            for heat in heats:
                params = {
                    'application': application,
                    'heating': heat,
                    'page': 'pg1'
                }
                text_html = await self.scrape_html(await self.get_url(params))
                page = await self.get_page(text_html)
                for i in range(1, page + 1):
                    params['page'] = 'pg' + str(i)
                    result_url = await self.get_url(params)
                    result_html = await self.scrape_html(result_url)
                    result_flood, result_house, result_price = await self.parse_html(result_html)
                    result_flood, result_house, result_price = await self.split_text((result_flood,
                                                                                      result_house, result_price))
                    data = (result_flood, result_house, result_price,applications[application], heats[heat])
                    await self.save_data_mongo_db(data)
                    # await self.save_data_csv(data)


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    spider = Crawler()
    loop.run_until_complete(spider.main())
