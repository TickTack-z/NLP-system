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
    word_return_df2 = pd.read_pickle('word_return_3month_full2.pickle')
    
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
            #sort_df = Tool.wordOfYear(word_return_df, filter_df, year+qtr).round(2)
            sort_df = Tool.wordOfYear(word_return_df2, filter_df, year+qtr).round(2)
        else:
            sort_df = Tool.wordOfQuarter(word_return_df, filter_df, year+qtr).round(2)

        sort_df['key_word'] = sort_df['key_word'].apply(lambda x: addHyperLink(year, qtr, x))
        pd.set_option('display.max_colwidth', -1)
        pd.set_option('display.width', None)


        output = ''
        output += r'<input class="form-control" id="myInput" type="text" placeholder="Search..">'
        output += HTML(sort_df.to_html(escape = False)).__html__().replace('<td><a ', '<td nowrap><a ').replace('class="dataframe"', 'class="table table-bordered table-striped"').replace(r'<tbody>', r'<tbody id="myTable">')
        
        output += r'''
<script>
$(document).ready(function(){
  $("#myInput").on("keyup", function() {
      var value = $(this).val().toLowerCase();
          $("#myTable tr").filter(function() {
                $(this).toggle($(this).text().toLowerCase().indexOf(value) > -1)
                    });
                      });
                      });
                      </script>

        '''


        context = {'year':year, 'qtr':qtr, 'output': output}
        return render(request, 'demo/home.html', context)
    else:
        print(word)
        #f =  Tool.plotSearchInterest(filter_df, word)
        tickers = Tool.searchWords(filter_df, word, year+qtr)
        #return_plot = Tool.returnPlot(word_return_df, word)
        
        if qtr=='':
            from_date = r'01/01/' + year
            to_date = r'12/31/' + year
        else:
            month = int(qtr[-1]) * 3
            month = str(month)

            from_date = (pd._libs.tslibs.period.Period(year + '-' + month,'M') - 6).strftime(r'%m/01/%Y')
            to_date = (pd._libs.tslibs.period.Period(year + '-' + month,'M') - 3 ).strftime(r'%m/%d/%Y')


        news= Tool.scrape_news_summaries(word, from_date, to_date)
        news_text = " ".join([(" ").join(k) for k in news])
        Tool.plotSenti(news_text)
        
        context = {'year':year, 'qtr':qtr, 'word': word, 'tickers': tickers, 'news':news, 'from_date': from_date, 'to_date': to_date}
        return render(request, 'demo/home.html', context)

def report(request):
    try:
        year = request.GET['year']
    except:
        return render(request, 'demo/report.html')

    qtr = request.GET['qtr']
    word = request.GET['word']
    ticker = request.GET['ticker']

    import json
    with open('ticker_to_cik.json', 'r') as json_file:
        ticker_to_cik = json.load(json_file)
    cik = ticker_to_cik[ticker]
    print(cik)

    some_text = Tool.generateTextForTicker(ticker, year, cik, qtr)

    context = {'year':year, 'qtr':qtr, 'word': word, 'tickers': ticker , 'ticker': ticker, 'text': some_text}

    return render(request, 'demo/report.html', context)

'''
def search_word(request):
    #year = request.GET['year']
    word= request.GET['word']
    print(word)
    
    return render(request, 'demo/word_search.html', {'year': word.replace(' ','')})
'''

    


