# pages/urls.py
from django.urls import path
from .views import homePageView, aboutPageView, JohnPageView,results, homePost

urlpatterns = [
    path('', homePageView, name='home'),
    path('about/', aboutPageView, name='about'),
    path('John/', JohnPageView, name='John'),
    path('homePost/', homePost, name='homePost'),
    path('<int:choice>/results/', results, name='results'),
    path('results/<int:choice>/<str:gmat>/', results, name='results'),
]