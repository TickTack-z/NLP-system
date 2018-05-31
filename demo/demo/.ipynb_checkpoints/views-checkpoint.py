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
from IPython.display import HTML


# http://yuji.wordpress.com/2013/01/30/django-form-field-in-initial-data-requires-a-fieldfile-instance/
class FakeField(object):
    storage = default_storage

fieldfile = FieldFile(None, FakeField, 'dummy.txt')

class HomePageView(TemplateView):
    template_name = 'demo/home.html'

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
        context = super(SearchTicker, self).get_context_data(**kwargs)
        return context
    
    
def search_ticker(request):
    #KQ = request.GET['KQ']
    word_return_df = pd.read_pickle('word_return_3month_full.pickle')
    
    filter_df = pd.read_pickle('filtered.pickle') 
    print(request)
    year = request.GET['year']
    qtr = request.GET['qtr']
    word = request.GET['word']
    
    if word == '':
        def addHyperLink(year, qtr, x):
            url = r'<a href="http://eqlnxwork1.panagora.com:8000/search_ticker/?year=' + str(year) +'&qtr=' + str(qtr)+'&word=' + x+ '"> ' + x + ' </a>'
            return url

        if qtr == '':
            sort_df = Tool.wordOfYear(word_return_df, filter_df, year+qtr).round(2)
        else:
            sort_df = Tool.wordOfQuarter(word_return_df, filter_df, year+qtr).round(2)

        sort_df['key_word'] = sort_df['key_word'].apply(lambda x: addHyperLink(year, qtr, x))
        pd.set_option('display.max_colwidth', -1)
        pd.set_option('display.width', None)
        context = {'year':year, 'qtr':qtr, 'output': HTML(sort_df.to_html(escape = False)).__html__().replace('<td><a ', '<td nowrap><a ')}
        return render(request, 'demo/home.html', context)
    else:
        print(word)
        Tool.plotSearchInterest(filter_df, 'sickle cell anemia')
        graphic = cStringIO.StringIO()
        canvas.print_png(graphic)
        
        context = {'year':year, 'qtr':qtr, 'word' : word, 'graphic': graphic}
        
        
        return render(request, 'demo/home.html', context)

'''
def search_word(request):
    #year = request.GET['year']
    word= request.GET['word']
    print(word)
    
    return render(request, 'demo/word_search.html', {'year': word.replace(' ','')})
'''