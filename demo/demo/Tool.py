
# coding: utf-8

# In[2]:


import pandas as pd
from functools import reduce
import ast
import datetime
import numpy as np


# In[2]:


def searchCompany(ticker):
    #return keyword list
    pass


# In[3]:


def generateFigure():
    pass


# In[ ]:


def load_word_list(new_path):
    word_list=[]
    for line in open(new_path):
        if line.strip()[0:1] != "#":
            for word in line.split():  # in case more than one per line
                word_list.append(word)
    return word_list


# In[ ]:


def load_phrase_list(new_path):
    word_list=[]
    for line in open(new_path):
        if line.strip()[0:1] != "#":
            word_list.append(line.replace('\n',''))
    return word_list


# In[ ]:


def filterByWord(merged_df, word_set):
    text = merged_df
    pd2 = pd.DataFrame(columns= list(text))
    for i in range(text.shape[0]):
        text_list = [k.lower() for k,n,j in ast.literal_eval(text["topic"][i])] + [n.lower() for k,n,j in ast.literal_eval(text["topic"][i])]
        text_list = [z for j in text_list for z in j.split()]
        text_list = set(text_list)
        if len(set(text_list) & word_set) == 0:
            continue
        pd2 = pd2.append(text.iloc[i])
    return pd2


# In[6]:


def simplify(df):
    for j in df.columns:
        if j == 'word:':
            continue
        elif j == 'search_interest':
            continue
        elif j == 'topic':
            continue    
        elif j == 'combined_words':
            continue
        else:        
            df[j] = df[j].apply(lambda x: [i for i,j in ast.literal_eval(x)] if pd.notnull(x) else [])


# In[ ]:


def combnineSameTicker(df):
    for j in df.columns:
        if j == 'word:':
            continue
        elif j == 'search_interest':
            continue
        elif j == 'topic':
            continue    
        elif j == 'combined_words':
            continue
        else:        
            df[j] = df[j].apply(lambda x: list(set(x)))


# In[ ]:


def combineTwoLines(df, idx1, idx2):
    pass
    


# In[ ]:


def isSimilar(word1, word2):
    word1_list = set(word1.split())
    word2_list = set(word2.split())
    #print(word1_list.intersection(word2_list))
    return len(word1_list.intersection(word2_list))>=2


# In[ ]:


def searchWords(df, word, quarter = ''):
    if word in df['word:'].tolist():
        idx = df.index[df['word:'] == word].tolist()[0]                
        if type(quarter) == type(1) or len(quarter) == 4:
            quarter = str(quarter)
            quarters = [quarter + i for i in ['QTR1', 'QTR2', 'QTR3', 'QTR4']]
            res = []
            for j in quarters:
                res += df.at[idx, j]
            return list(set(res))
        elif quarter != '':
            return df.at[idx, quarter]
        else:
            res = []
            for j in df.columns:
                if j == 'word:':
                    continue
                elif j == 'search_interest':
                    continue
                elif j == 'topic':
                    continue    
                elif j == 'combined_words':
                    continue
                else:        
                    res += df.at[idx, j]
            return list(set(res))
    else:
        for idx in df.index:
            if word in df.at[idx, 'word:']:
                word = df.at[idx, 'word:']
                return (word, searchWords(df, word, quarter))
        return False


# In[ ]:


def searchTicker(df, ticker, quarter = ''):
    if ticker in df.index.tolist():
        idx = ticker        
        if type(quarter) == type(1) or len(quarter) == 4:
            quarter = str(quarter)
            quarters = [quarter + i for i in ['QTR1', 'QTR2', 'QTR3', 'QTR4']]
            res = []
            for j in quarters:
                res += df.at[idx, j]
            return list(set(res))
        elif quarter != '':
            return df.at[idx, quarter]
        else:
            res = []
            for j in df.columns:
                if j == 'word:':
                    continue
                elif j == 'search_interest':
                    continue
                elif j == 'topic':
                    continue    
                elif j == 'combined_words':
                    continue
                else:        
                    res += df.at[idx, j]
            return list(set(res))


# In[9]:


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)


# In[ ]:


def search2Ticker(df, ticker1, ticker2, quarter = ''):
    res_dict = {ticker1: [], ticker2: [], 'common phrase': []}
    if ticker1 in df.index.tolist():
        idx = ticker1
        if type(quarter) == type(1) or len(quarter) == 4:
            quarter = str(quarter)
            quarters = [quarter + i for i in ['QTR1', 'QTR2', 'QTR3', 'QTR4']]
            res = []
            for j in quarters:
                res += df.at[idx, j]
            res_dict[ticker1] = res
        elif quarter != '':
            res_dict[ticker1] = df.at[idx, quarter]
        else:
            res = []
            for j in df.columns:
                if j == 'word:':
                    continue
                elif j == 'search_interest':
                    continue
                elif j == 'topic':
                    continue    
                elif j == 'combined_words':
                    continue
                else:        
                    res += df.at[idx, j]
            res_dict[ticker1] = res
    if ticker2 in df.index.tolist():
        idx = ticker2
        if type(quarter) == type(1) or len(quarter) == 4:
            quarter = str(quarter)
            quarters = [quarter + i for i in ['QTR1', 'QTR2', 'QTR3', 'QTR4']]
            res = []
            for j in quarters:
                res += df.at[idx, j]
            res_dict[ticker2] = res
        elif quarter != '':
            res_dict[ticker2] = df.at[idx, quarter]
        else:
            res = []
            for j in df.columns:
                if j == 'word:':
                    continue
                elif j == 'search_interest':
                    continue
                elif j == 'topic':
                    continue    
                elif j == 'combined_words':
                    continue
                else:        
                    res += df.at[idx, j]
            res_dict[ticker2] = res
    res_dict['common phrase'] = list(set(res_dict[ticker2]).intersection(set(res_dict[ticker1])))
    
    intersection_of_2 = list(set(res_dict[ticker2]+res_dict[ticker1]))
    v1 = [1 if k in res_dict[ticker2] else 0 for k in intersection_of_2]
    v2 = [1 if k in res_dict[ticker1] else 0 for k in intersection_of_2]
    
    return (res_dict, angle_between(v1, v2))


# In[ ]:


def getTopicOfPhrase(df, word):
    return df.at[df.loc[df['word:'] == word, 'topic'].index[0], 'topic']


# In[2]:


def plotSearchInterest(filter_df, word):
    class MyCalendar():
        def __init__(self, startDate, endDate):
            """
            The input parameters should be dates. Moreover, the start date should be lower or
            equal to the end date.
            """
            if type(startDate) != datetime.date or type(endDate) != datetime.date:
                raise StandardError('The parameters in input should be dates!')
            if startDate > endDate:
                raise StandardError('The start date should be lower or equal than the end date!')
            self.__startDate = startDate
            self.__endDate = endDate

        def getWeekDays(self):
            """
            Return a list of week days in a date range.
            """
            dateRange = (self.__endDate +datetime.timedelta(days=1)-self.__startDate).days
            # According to http://docs.python.org/library/datetime.html, the weekday function will return a value from
            # 0(Monday) to 6(Sunday).
            FRIDAY = 4
            days =  [(self.__startDate + datetime.timedelta(days=i)) for i in range(dateRange) if (self.__startDate+datetime.timedelta(days=i)).weekday() <= FRIDAY]
            return days

        def getWeekDaysStr(self):
            cal = [i.strftime("%Y%m%d") for i in self.getWeekDays()]
            cal.sort(reverse = True)
            return cal

        def getOneMonthYearMonthDayList(self, downloaded_year_month_day_list):
            #assume a month has 21 working days
            return downloaded_year_month_day_list[:21]

        def getWeekDaysMonthStr(self):
            cal = list(set([i.strftime("%Y%m") for i in self.getWeekDays()]))
            cal.sort(reverse = True)
            return cal
    
    data_list = [i for i,j in ast.literal_eval(filter_df.at[filter_df.loc[filter_df['word:'] == word, 'topic'].index[0], 'search_interest'])]
    
    
    fromDate = datetime.date(2004,1,1)
    toDate = datetime.date(2018,4,22)
    cal = MyCalendar(fromDate, toDate)
    xticks = cal.getWeekDaysMonthStr()
    xticks.reverse()
    
    import matplotlib
    matplotlib.use('agg')
    import pylab
    some_var = pylab.figure(1, figsize=(22, 8))
    xaxis = range(len(xticks))
    pylab.xticks(xaxis[::6], xticks[::6], rotation = 60, fontsize = 13)
    pylab.xlabel("year_month")
    pylab.ylabel("search interest")
    pylab.plot(xaxis,data_list,"g")        
    pylab.show()
    
    import io
    import os
    f = io.BytesIO()
    
    try:
        os.remove(r'common_static/foo.png')
    except OSError:
        pass
    pylab.savefig(r'common_static/foo.png', format = 'png')
    pylab.clf()
    #return f.getvalue()
    


# In[ ]:


def generateCluster(filter_df, quarter):    
    ticker_list = []   
    word_list = []
    for idx in filter_df.index:
        word = filter_df.at[idx, 'word:']
        if len(filter_df.at[idx, quarter]) >0:
            word_list.append(word)
            ticker_list += filter_df.at[idx, quarter]
    
    ticker_list = list(set(ticker_list))
    word_list =  list(set(word_list))
    
    new_df = pd.DataFrame(0, columns = ticker_list, index = word_list)   
    
    for idx in filter_df.index:
        word = filter_df.at[idx, 'word:']
        if len(filter_df.at[idx, quarter]) >0:            
            for ticker in filter_df.at[idx, quarter]:
                new_df.at[word, ticker] = 1
    new_df.to_csv("matrix.csv",sep='\t')


# In[ ]:


'''
def wordOfQuarter(new_df, qtr):
    return new_df.sort_values(qtr, ascending=False)['word:'].tolist()'''


# In[ ]:


def wordOfQuarter(new_df, filter_df, qtr):
    temp_df = new_df.copy()
    
    def convertColumns2yearmonth(columns):
        year = int(columns[:4])
        qtr = columns[-4:]
        month = int(qtr[-1]) * 3
        return pd._libs.tslibs.period.Period(str(year) + '-' + str(month),'M') - 3
    
    temp_df['mean'] = 0.0
    temp_df['median'] = 0.0
    temp_df['std'] = 0.0
    for idx in temp_df.index:
        columns = qtr     

        year_month = convertColumns2yearmonth(columns)
        
        temp_df.at[idx, 'mean'] = np.mean(temp_df.at[idx, columns]) if len(temp_df.at[idx, columns])>=3 else np.nan
        temp_df.at[idx, 'median'] = np.median(temp_df.at[idx, columns]) if len(temp_df.at[idx, columns])>=3 else np.nan
        temp_df.at[idx, 'std'] = np.std(temp_df.at[idx, columns]) if len(temp_df.at[idx, columns])>=3 else np.nan
        
    
    temp_df2 = temp_df.sort_values('median', ascending=False)[['word:' , 'mean', 'median', 'std']].dropna().merge(filter_df[['word:',qtr]], left_on = ['word:'], right_on = ['word:'])
    temp_df2.columns = ['key_word', 'annualized_return_percentage_mean','median','std', 'tickers']
    temp_df2['annualized_return_percentage_mean'] = temp_df2['annualized_return_percentage_mean'].apply(lambda x: ((x/100.0+1)**4-1)*100)
    temp_df2['median'] = temp_df2['median'].apply(lambda x: ((x/100.0+1)**4-1)*100)    
    return temp_df2

    
            


# In[17]:


def wordOfYear(new_df, filter_df, year):
    year = int(year)
    temp_df = new_df.copy()
    temp_filter_df = filter_df.copy()
    
    def convertColumns2yearmonth(columns):
        year = int(columns[:4])
        qtr = columns[-4:]
        month = int(qtr[-1]) * 3
        return pd._libs.tslibs.period.Period(str(year) + '-' + str(month),'M') - 3
    
    
    temp_df[str(year)+'QTR1'] = temp_df[str(year)+'QTR1'].apply(lambda x: x[:])
    temp_filter_df[str(year)+'QTR1'] = temp_filter_df[str(year)+'QTR1'].apply(lambda x: x[:])
        
    temp_df['mean'] = 0.0
    temp_df['median'] = 0.0
    temp_df['std'] = 0.0
    
    for idx in temp_df.index:
        for qtr in [str(year)+k for k in ['QTR1', 'QTR2','QTR3', 'QTR4']]:            
            columns = qtr

            year_month = convertColumns2yearmonth(columns)

            temp_df.at[idx, str(year)+'QTR1'] += new_df.at[idx, columns]
            temp_filter_df.at[idx, str(year)+'QTR1'] += filter_df.at[idx, columns]
    qtr = columns = str(year)+'QTR1'
    for idx in temp_df.index:
        temp_df.at[idx, 'mean'] = np.mean(temp_df.at[idx, columns]) if len(temp_df.at[idx, columns])>=9 else np.nan
        temp_df.at[idx, 'median'] = np.median(temp_df.at[idx, columns]) if len(temp_df.at[idx, columns])>=9 else np.nan
        temp_df.at[idx, 'std'] = np.std(temp_df.at[idx, columns]) if len(temp_df.at[idx, columns])>=9 else np.nan
    
    temp_df2 = temp_df.sort_values('median', ascending=False)[['word:' , 'mean', 'median' ,'std']].dropna().merge(temp_filter_df[['word:',qtr]], left_on = ['word:'], right_on = ['word:'])
    temp_df2.columns = ['key_word', 'return_percentage_mean','median','std', 'tickers']
    temp_df2['return_percentage_mean'] = temp_df2['return_percentage_mean'].apply(lambda x: ((x/100.0+1)**4-1)*100)
    temp_df2['median'] = temp_df2['median'].apply(lambda x: ((x/100.0+1)**4-1)*100)    
    temp_df2['tickers'] = temp_df2['tickers'].apply(lambda x: list(set(x)))
    return temp_df2


