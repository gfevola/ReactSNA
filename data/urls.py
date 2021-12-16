#completeversion/data/urls
from rest_framework import routers
from django.urls import path, include
from .views import DataView, DataViewPost


#router = routers.DefaultRouter()
#router.register(r'data',DataView,'data')

urlpatterns = [
   # path('',include(router.urls)),
    path('data',DataView.as_view()),
    path('datapost',DataViewPost.as_view())
]