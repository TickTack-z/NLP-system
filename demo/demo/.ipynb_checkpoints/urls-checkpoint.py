# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.conf.urls import url

from .views import HomePageView, FormWithFilesView, search_ticker, SearchTicker


urlpatterns = [
    url(r'^$', HomePageView.as_view(), name='home'),
    url(r'^form_with_files$', FormWithFilesView.as_view(), name='form_with_files'),
    url(r'^search_ticker/', search_ticker, name='search_ticker'),
]