import json
import pyhdb
# import numpy as np
from datetime import datetime, timedelta
from .mlalgo.model_v2 import predict, _predict

class DataProcessHandler(object):

    def __init__(self, request):
        self.request = request
        self.connection = pyhdb.connect(host="10.58.114.58",
                                   port=30015,
                                   user="SYSTEM",
                                   password="manager")
        self.object_type = 'POR'
        self.from_date = '20110101'
        self.to_date = '20170331'
        self.fields = 'Price,Quantity'

    def parse_parameters(self):
        meta_keys = self.request.META.keys()
        print('meta_keys received: {}'.format(meta_keys))
        if 'object' in meta_keys:
            self.object_type = self.request.META['object']
        if 'HTTP_FROM' in meta_keys:
            self.from_date = self.request.META['HTTP_FROM']
        if 'HTTP_TO' in meta_keys:
            self.to_date = self.request.META['HTTP_TO']
        if 'fields' in meta_keys:
            self.fields = meta_keys['fields']

        if not self.check_parameters():
            raise RuntimeError("Invalid parameter!")

    def check_parameters(self):
        # Todo
        return True

    """
    'SELECT T1."DATETIMESTAMP", IFNULL("Price", 0), IFNULL("Quantity", 0) \
                        FROM HACKATHON5.RDR1 T0 \
                   RIGHT OUTER JOIN "_SYS_BI"."M_TIME_DIMENSION" T1 \
                   ON T0."DocDate" = T1."DATETIMESTAMP" \
                   WHERE T1."DATETIMESTAMP" BETWEEN \' 20110101\' AND \'20170331\'  \
                    order by T1."DATETIMESTAMP"'
    """
    def spell_sql(self):
        sql = 'SELECT TO_NVARCHAR(T0."DATETIMESTAMP", \'YYYYMMDD\'), '
        # all_fields = self.fields.split(',')
        # for f in all_fields:
        #     sql = sql + 'SUM(IFNULL("{}", 0)) '.format(f)
        #     if f != all_fields[-1]:
        #         sql = sql + ','
        sql = sql + 'SUM(IFNULL(T1."Price", 0)), SUM(IFNULL(T2."Quantity", 0)) '
        sql = sql + 'FROM "_SYS_BI"."M_TIME_DIMENSION" T0 '
        sql = sql + 'LEFT OUTER JOIN HACKATHON5.POR1 T1 '
        sql = sql + 'ON T1."DocDate" = T0."DATETIMESTAMP" '
        sql = sql + 'LEFT OUTER JOIN HACKATHON5.RDR1 T2 '
        sql = sql + 'ON T2."DocDate" = T0."DATETIMESTAMP" '

        sql = sql + 'WHERE T0."DATETIMESTAMP" BETWEEN \' {}\' AND \'{}\' '.format(self.from_date, self.to_date)
        sql = sql + 'GROUP BY T0."DATETIMESTAMP" '
        sql = sql + 'ORDER BY T0."DATETIMESTAMP" '

        return sql

    def fetch_all(self):
        cursor = self.connection.cursor()
        sql = self.spell_sql()
        print('sql executed: ')
        print(sql)

        try:
            cursor.execute(sql)
            all_result = cursor.fetchall()
        except Exception as exp:
            print('Exception!')
            print(exp)
            # self.connection.close()
            return None

        # self.connection.close()
        return all_result

    def fetch_all_to_json(self):
        all_result = self.fetch_all()
        result_dict = dict()
        for (docdate, price, quantity) in all_result:
            item = dict()
            item['Price'] = str(price)
            item['Quantity'] = str(quantity)
            item['Date'] = docdate
            result_dict[docdate] = item

        return json.dumps(result_dict, sort_keys=True)

class ForcastDataHandler(object):
    def __init__(self, request):
        self.request = request
        self.purchase_price = list()
        self.sales_quantity = list()

    def forcast(self, mode='full'):
        mode = 'full'
        KEY = 'HTTP_DATARANGE'
        print('in request:')
        print(self.request.META.keys())
        if KEY in self.request.META.keys():
            print('find key DATARANGE')
            range = self.request.META[KEY]
            if range == 'all':
                mode = 'full'
            else:
                mode = 'year'

        # for i in range(0, 90):
        #     purchase_price.append(float(i + np.random.random() * 100))
        # for j in range(0, 90):
        #     sales_quantity.append(float(j + np.random.random() * 100))

        self.purchase_price, self.sales_quantity = predict(mode=mode)
        return self.purchase_price, self.sales_quantity

    def push_data_in_json(self):
        purchase_price, sales_quantity = self.forcast()
        date_span = min(len(purchase_price), len(sales_quantity))
        print('date span is {}'.format(date_span))

        start_date = datetime.strptime('20170401', '%Y%m%d')
        temp = start_date

        ret = dict()
        for i in range(0, date_span):
            price = round(purchase_price[i], 2)
            quantity = int(sales_quantity[i])
            date = temp.strftime('%Y%m%d')
            item = dict()
            item['Date'] = date
            item['Price'] = str(price)
            item['Quantity'] = str(quantity)
            ret[date] = item

            temp = start_date + timedelta(days=i + 1)

        print(ret)
        return json.dumps(ret, sort_keys=True)


class RecommendHandler(object):
    def __init__(self, request, forecast_handler=None):
        self.request = request
        self.forecast = forecast_handler
        self.period = 7
        self.weekdays = list()
        self.confidence = 1.0

    def parse_parameters(self):
        key = 'HTTP_PERIOD'
        if key in self.request.META.keys():
            print('find key {}'.format(key))
            self.period = int(self.request.META[key])
        print('In recommend : period = {}'.format(self.period))

        key = 'HTTP_WEEKDAYS'
        if key in self.request.META.keys():
            print('find key {}'.format(key))
            weekdays = self.request.META[key]
            print('value {}'.format(weekdays))
            days = weekdays.split(',')
            for day in days:
                self.weekdays.append(int(day))

        if len(self.weekdays) == 0:
            self.weekdays = list().extend([0,1,2,3,4,5,6,7])
        print(' weekdays = {}'.format(self.weekdays))

        key = 'HTTP_CONFIDENCE'
        if key in self.request.META.keys():
            print('find key {} '.format(key))
            self.confidence = float(self.request.META[key])
        print(' confidence = {}'.format(self.confidence))

    def push_recommend_data_in_json(self):
        self.parse_parameters()
        profit, recommend_quantity = self.recommend()
        if profit is None:
            return json.dumps(dict(), sort_keys=True)

        date_span = len(recommend_quantity)
        start_date = datetime.strptime('20170401', '%Y%m%d')
        temp = start_date

        ret = dict()
        for i in range(0, date_span):
            quantity = int(recommend_quantity[i])
            date = temp.strftime('%Y%m%d')
            item = dict()
            item['Date'] = date
            item['Recommend'] = str(quantity)
            ret[date] = item

            temp = start_date + timedelta(days=i + 1)

        return json.dumps(ret, sort_keys=True)

    def recommend(self):
        from .mlalgo.recommendation import recommend
        if self.forecast is None:
            return None, None

        profit, recommend_quantity = recommend(self.forecast.sales_quantity,
                                               self.forecast.purchase_price,
                                               inventory_period=self.period,
                                               alpha=self.confidence,
                                               purchase_days=self.weekdays)
        return profit, recommend_quantity


