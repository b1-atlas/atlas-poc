from django.views.generic.base import View
from django.http import HttpResponse, HttpRequest

from .handlers import DataProcessHandler
from .handlers import ForcastDataHandler
from .handlers import RecommendHandler


global_forecast_handler = None

class FetchDocumentDataView(View):
    def dispatch(self, request, *args, **kwargs):
        handler = DataProcessHandler(request)
        handler.parse_parameters()
        text = handler.fetch_all_to_json()

        response = HttpResponse(text)
        response["Access-Control-Allow-Methods"] = "GET, PUT, POST, DELETE, OPTIONS"
        response["Access-Control-Allow-Headers"] = "X-Requested-With, Content-type, " \
                                                   "Accept, Content-Length, Authorization," \
                                                   "from, to"
        response["Access-Control-Allow-Origin"] = "*"
        response["Access-Control-Allow-Credentials"] = "true"

        return response


class ForcastView(View):
    def dispatch(self, request, *args, **kwargs):

        handler = ForcastDataHandler(request)
        global  global_forecast_handler
        global_forecast_handler = handler
        ret = handler.push_data_in_json()

        response = HttpResponse(ret)
        response["Access-Control-Allow-Methods"] = "GET, PUT, POST, DELETE, OPTIONS"
        response["Access-Control-Allow-Headers"] = "X-Requested-With, Content-type, " \
                                                   "Accept, Content-Length, Authorization," \
                                                   "DATARANGE"
        response["Access-Control-Allow-Origin"] = "*"
        response["Access-Control-Allow-Credentials"] = "true"

        return response

    # def options(self, request, *args, **kwargs):
    #     method = request.META.get('Access-Control-Allow-Method')
    #     origin = request.META.get('Access-Control-Allow-Origin')
    #
    #     response = HttpResponse()
    #     response['Access-Control-Allow-Method'] = method
    #     response['Access-Control-Allow-Origin'] = origin
    #
    #     print('server response in options method')
    #     print(response)
    #     return response


class RecommendView(View):
    def dispatch(self, request, *args, **kwargs):

        handler = RecommendHandler(request, global_forecast_handler)
        ret = handler.push_recommend_data_in_json()

        response = HttpResponse(ret)
        response["Access-Control-Allow-Methods"] = "GET, PUT, POST, DELETE, OPTIONS"
        response["Access-Control-Allow-Headers"] = "X-Requested-With, Content-type, " \
                                                   "Accept, Content-Length, Authorization," \
                                                   "PERIOD, WEEKDAYS, CONFIDENCE"
        response["Access-Control-Allow-Origin"] = "*"
        response["Access-Control-Allow-Credentials"] = "true"

        print('In RecommendView return: {}'.format(ret))
        return response
