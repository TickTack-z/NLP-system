
# coding: utf-8

# In[1]:


import pandas as pd
from functools import reduce
import ast
import datetime


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
    res_dict['common phrase'] = list(set(res_dict[ticker2] + res_dict[ticker2]))
    return res_dict


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
    
    import pylab
    pylab.figure(1, figsize=(22, 8))
    xaxis = range(len(xticks))
    pylab.xticks(xaxis, xticks, rotation = 60, fontsize = 7)
    pylab.xlabel("year_month")
    pylab.ylabel("search interest")
    pylab.plot(xaxis,data_list,"g")        
    pylab.show()


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


def wordOfQuarter(new_df, qtr):
    return new_df.sort_values(qtr, ascending=False)['word:'].tolist()

