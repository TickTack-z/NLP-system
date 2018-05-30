# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.core.files.storage import default_storage

from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from django.db.models.fields.files import FieldFile
from django.views.generic import FormView
from django.views.generic.base import TemplateView
from django.contrib import messages
from django.shortcuts import render

from .forms import ContactForm, FilesForm, ContactFormSet
import os
import demo.Tool as Tool
import pandas as pd


# http://yuji.wordpress.com/2013/01/30/django-form-field-in-initial-data-requires-a-fieldfile-instance/
class FakeField(object):
    storage = default_storage

fieldfile = FieldFile(None, FakeField, 'dummy.txt')

class HomePageView(TemplateView):
    template_name = 'demo/home.html'

    def get_context_data(self, **kwargs):
        context = super(HomePageView, self).get_context_data(**kwargs)
        #messages.info(self.request, 'hello http://example.com')
        return context


class FormWithFilesView(FormView):
    template_name = 'demo/form_with_files.html'
    form_class = FilesForm

    def get_context_data(self, **kwargs):
        context = super(FormWithFilesView, self).get_context_data(**kwargs)
        context['layout'] = self.request.GET.get('layout', 'vertical')
        return context

    def get_initial(self):
        return {
            'file4': fieldfile,
        }

    
class SearchTicker(TemplateView):
    template_name = 'demo/home.html'
    def get_context_data(self, **kwargs):
        context = super(PaginationView, self).get_context_data(**kwargs)
        print(context)
        return context
    
    
def search_ticker(request):
    #KQ = request.GET['KQ']
    filter_df = pd.read_pickle('filtered.pickle')
    ticker_word_df = pd.read_pickle('ticker_word_df.pickle')
    word_return_df = pd.read_pickle('word_return_3month.pickle')
    print(request)
    ticker = request.GET['ticker']
    ticker2 = request.GET['ticker2']
    print(os.getcwd())
    print(Tool.searchTicker(ticker_word_df, ticker, '2015'))
    print(ticker2)
    return render(request, 'demo/home.html', {'ticker': ticker, 'ticker2': ticker2})