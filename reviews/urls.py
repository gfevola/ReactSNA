#completeversion/data/urls
from rest_framework import routers
from django.urls import path, include
from .views import ReviewViewPost, ReviewTemplate, ReviewModelsPost, ReviewModelBigram_View, DeleteModelsPost


#router = routers.DefaultRouter()
#router.register(r'models',ReviewModelsPost,'model')

urlpatterns = [
    #path('',include(router.urls)),
    path('review/',ReviewViewPost.as_view()),
    path('upload/',ReviewTemplate, name="upload-review"),
    path('models/',ReviewModelsPost.as_view()),
    path('bigram/',ReviewModelBigram_View.as_view()),
    path('delete/',DeleteModelsPost.as_view()),
]