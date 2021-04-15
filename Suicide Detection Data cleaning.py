#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd
import numpy as np

import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import re
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
tok = WordPunctTokenizer()


# In[3]:


depressive_tweets_df = pd.read_csv('D:/ptnt/Suicide Detection/depression/depressive_unigram_tweets.csv')


# In[4]:


depressive_tweets_df.head()


# In[6]:


depression_tweets_df = pd.read_csv('D:/ptnt/Suicide Detection/depression/depressive_unigram_tweets.csv')


# In[7]:


depression_tweets_df.head()


# In[ ]:





# In[8]:


depression_tweets_df.drop(['date', 'timezone', 'username', 'name', 'conversation_id', 'created_at', 'user_id', 'place', 'likes_count', 'link', 'retweet', 'quote_url', 'video', 'user_rt_id', 'near', 'geo', 'mentions', 'urls', 'photos', 'replies_count', 'retweets_count'], axis = 1, inplace = True)


# In[9]:


depressed_tweets_df = pd.read_csv('D:/ptnt/Suicide Detection/depressed/tweets.csv')


# In[10]:


depressed_tweets_df.head()


# In[11]:


depressed_tweets_df.drop(['date', 'timezone', 'username', 'name', 'conversation_id', 'created_at', 'user_id', 'place', 'likes_count', 'link', 'retweet', 'quote_url', 'video', 'user_rt_id', 'near', 'geo', 'mentions', 'urls', 'photos', 'replies_count', 'retweets_count'], axis = 1, inplace = True)


# In[12]:


depressed_tweets_df.head()


# In[13]:


hopeless_tweets_df = pd.read_csv('D:/ptnt/Suicide Detection/hopeless/tweets.csv')


# In[14]:


hopeless_tweets_df.head()


# In[15]:


hopeless_tweets_df.drop(['date', 'timezone', 'username', 'name', 'conversation_id', 'created_at', 'user_id', 'place', 'likes_count', 'link', 'retweet', 'quote_url', 'video', 'user_rt_id', 'near', 'geo', 'mentions', 'urls', 'photos', 'replies_count', 'retweets_count'], axis = 1, inplace = True)


# In[ ]:





# In[16]:


hopeless_tweets_df.head()


# In[17]:


lonely_tweets_df = pd.read_csv('D:/ptnt/Suicide Detection/lonely/tweets.csv')


# In[19]:


lonely_tweets_df.head()


# In[ ]:





# In[20]:


lonely_tweets_df.drop(['date', 'timezone', 'username', 'name', 'conversation_id', 'created_at', 'user_id', 'place', 'likes_count', 'link', 'retweet', 'quote_url', 'video', 'user_rt_id', 'near', 'geo', 'mentions', 'urls', 'photos', 'replies_count', 'retweets_count'], axis = 1, inplace = True)


# In[21]:


lonely_tweets_df.head()


# In[23]:


antidepressant_tweets_df = pd.read_csv('D:/ptnt/Suicide Detection/antidepressant/tweets.csv')


# In[24]:


antidepressant_tweets_df.head()


# In[ ]:





# In[25]:


antidepressant_tweets_df.drop(['date', 'timezone', 'username', 'name', 'conversation_id', 'created_at', 'user_id', 'place', 'likes_count', 'link', 'retweet', 'quote_url', 'video', 'user_rt_id', 'near', 'geo', 'mentions', 'urls', 'photos', 'replies_count', 'retweets_count'], axis = 1, inplace = True)


# In[26]:


antidepressant_tweets_df.head()


# In[27]:


antidepressants_tweets_df = pd.read_csv('D:/ptnt/Suicide Detection/antidepressants/tweets.csv')


# In[29]:


antidepressants_tweets_df.head()


# In[30]:


antidepressants_tweets_df.drop(['date', 'timezone', 'username', 'name', 'conversation_id', 'created_at', 'user_id', 'place', 'likes_count', 'link', 'retweet', 'quote_url', 'video', 'user_rt_id', 'near', 'geo', 'mentions', 'urls', 'photos', 'replies_count', 'retweets_count'], axis = 1, inplace = True)


# In[31]:


antidepressants_tweets_df.head()


# In[32]:


suicide_tweets_df = pd.read_csv('D:/ptnt/Suicide Detection/suicide/tweets.csv')


# In[33]:


suicide_tweets_df.head()


# In[34]:


suicide_tweets_df.drop(['date', 'timezone', 'username', 'name', 'conversation_id', 'created_at', 'user_id', 'place', 'likes_count', 'link', 'retweet', 'quote_url', 'video', 'user_rt_id', 'near', 'geo', 'mentions', 'urls', 'photos', 'replies_count', 'retweets_count'], axis = 1, inplace = True)


# In[35]:


suicide_tweets_df.head()


# In[37]:


df_row_reindex = pd.concat([depression_tweets_df, hopeless_tweets_df, lonely_tweets_df, antidepressant_tweets_df, antidepressants_tweets_df, suicide_tweets_df], ignore_index=True)


# In[38]:


df_row_reindex


# In[39]:


df = df_row_reindex


# In[40]:


depressive_twint_tweets_df = df_row_reindex


# In[41]:


depressive_twint_tweets_df.head()


# In[42]:


depressive_twint_tweets_df = df.drop_duplicates()


# In[43]:


depressive_twint_tweets_df


# In[44]:


export_csv = depressive_twint_tweets_df.to_csv(r'D:/ptnt/Suicide Detection/depressive_unigram_tweets_final.csv')


# In[45]:


pd.read_csv('D:/ptnt/Suicide Detection/depressive_unigram_tweets_final.csv')


# In[49]:


import nltk
nltk.download(['punkt','stopwords'])
from nltk.corpus import stopwords
stopwords = stopwords.words('english')


# In[50]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


# In[52]:


df2 = pd.read_csv('D:/ptnt/Suicide Detection/depressive_unigram_tweets_final.csv')


# In[53]:


df2.head()


# In[65]:


df2.isnull().any().any()


# In[74]:


df2.info(null_counts=True)


# In[75]:


df_new = df2[df2['tweet'].notnull()]


# In[76]:


df_new.info(null_counts=True)


# In[77]:


df_new.isnull().any().any()


# In[78]:


df_new['clean_tweet'] = df_new['tweet'].apply(lambda x: ' '.join([item for item in x.split() if item not in stopwords]))


# In[79]:


df_new.head()


# In[80]:


df_new['vader_score'] = df_new['clean_tweet'].apply(lambda x: analyzer.polarity_scores(x)['compound'])


# In[81]:


df_new.head()


# In[84]:


positive_num = len(df_new[df_new['vader_score'] >=0.05])
negative_num = len(df_new[df_new['vader_score']<0.05])


# In[85]:


positive_num, negative_num


# In[86]:


df_new['vader_sentiment_label']= df_new['vader_score'].map(lambda x:int(1) if x>=0.05 else int(0))


# In[87]:


df_new.head()


# In[88]:


df_new.loc[df_new['vaderReviewScore'] >=0.00,"vaderSentimentLabel"] = 1


# In[89]:


df_new.drop(['Unnamed: 0.1', 'id', 'time', 'tweet', ], axis = 1, inplace = True)


# In[90]:


df_new = df_new[['Unnamed: 0', 'vader_sentiment_label', 'vader_score', 'clean_tweet']]


# In[91]:


df_new.head()


# In[92]:


positive_num = len(df_new[df_new['vader_score'] >=0.05])
neutral_num = len(df_new[(df_new['vader_score'] >-0.05) & (df_new['vader_score']<0.05)])
negative_num = len(df_new[df_new['vader_score']<=-0.05])


# In[93]:


positive_num,neutral_num, negative_num


# In[94]:


df_new.to_csv('D:/ptnt/Suicide Detection/vader_processed_final.csv')


# In[95]:


import nltk
from nltk.corpus import stopwords
import re
from nltk import bigrams
import networkx as nx


# In[112]:


import seaborn as sns


# In[113]:


import warnings
warnings.filterwarnings("ignore")


sns.set(font_scale=1.5)
sns.set_style("whitegrid")


# In[114]:


df_new['text'] = df_new['clean_tweet']


# In[115]:


df_new['text']


# In[116]:


def remove_url(txt):
    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())


# In[117]:


all_tweets_no_urls = [remove_url(tweet) for tweet in df_new['text']]
all_tweets_no_urls[:5]


# In[118]:


lower_case = [word.lower() for word in df_new['text']]
sentences = df_new['text']


# In[119]:


all_tweets_no_urls[0].split()


# In[120]:


all_tweets_no_urls[0].lower().split()


# In[121]:


words_in_tweet = [tweet.lower().split() for tweet in all_tweets_no_urls]
words_in_tweet[:2]


# In[125]:


import itertools
import collections


# In[126]:


all_words_no_urls = list(itertools.chain(*words_in_tweet))
counts_no_urls = collections.Counter(all_words_no_urls)


# In[132]:


counts_no_urls.most_common(20)


# In[133]:


clean_tweets_no_urls = pd.DataFrame(counts_no_urls.most_common(20),
                             columns=['words', 'count'])

clean_tweets_no_urls.head()


# In[134]:


fig, ax = plt.subplots(figsize=(10, 10))

# Plot horizontal bar graph
clean_tweets_no_urls.sort_values(by='count').plot.barh(x='words',
                      y='count',
                      ax=ax,
                      color="purple")

ax.set_title("Common Words Found in Tweets (Including All Words)")

plt.show()


# In[135]:


stop_words = set(stopwords.words('english'))

# View a few words from the set
list(stop_words)[0:15]


# In[139]:


words_in_tweet[11]


# In[140]:


tweets_nsw = [[word for word in tweet_words if not word in stop_words]
              for tweet_words in words_in_tweet]

tweets_nsw[11]


# In[143]:


all_words_nsw = list(itertools.chain(*tweets_nsw))

counts_nsw = collections.Counter(all_words_nsw)

counts_nsw.most_common(20)


# In[144]:


clean_tweets_nsw = pd.DataFrame(counts_nsw.most_common(20),
                             columns=['words', 'count'])

fig, ax = plt.subplots(figsize=(10, 10))

# Plot horizontal bar graph
clean_tweets_nsw.sort_values(by='count').plot.barh(x='words',
                      y='count',
                      ax=ax,
                      color="purple")

ax.set_title("Common Words Found in Tweets (Without Stop Words)")

plt.show()


# In[145]:


collection_words = ['im', 'de', 'like', 'one']
tweets_nsw_nc = [[w for w in word if not w in collection_words]
                 for word in tweets_nsw]


# In[146]:


tweets_nsw[11]


# In[147]:


tweets_nsw_nc[11]


# In[148]:


# Flatten list of words in clean tweets
all_words_nsw_nc = list(itertools.chain(*tweets_nsw_nc))

# Create counter of words in clean tweets
counts_nsw_nc = collections.Counter(all_words_nsw_nc)

counts_nsw_nc.most_common(20)


# In[150]:


len(counts_nsw_nc)


# In[152]:


from statistics import *


# In[153]:


mean(counts_nsw_nc)


# In[155]:


clean_tweets_ncw = pd.DataFrame(counts_nsw_nc.most_common(20),
                             columns=['words', 'count'])
clean_tweets_ncw.head()


# In[157]:


fig, ax = plt.subplots(figsize=(10, 10))

# Plot horizontal bar graph
clean_tweets_ncw.sort_values(by='count').plot.barh(x='words',
                      y='count',
                      ax=ax,
                      color="purple")

ax.set_title("Common Words Found in Tweets (Without Stop or Collection Words)")

plt.show()


# In[159]:


from nltk import bigrams

# Create list of lists containing bigrams in tweets
terms_bigram = [list(bigrams(tweet)) for tweet in tweets_nsw_nc]

# View bigrams for the first tweet
terms_bigram[11]


# In[161]:


# Flatten list of bigrams in clean tweets
bigrams = list(itertools.chain(*terms_bigram))

# Create counter of words in clean bigrams
bigram_counts = collections.Counter(bigrams)

bigram_counts.most_common(25)


# In[162]:


bigram_df = pd.DataFrame(bigram_counts.most_common(25),
                             columns=['bigram', 'count'])

bigram_df


# In[163]:


bigram_df = pd.DataFrame(bigram_counts.most_common(25),
                             columns=['bigram', 'count'])

bigram_df


# In[164]:


# Create dictionary of bigrams and their counts
d = bigram_df.set_index('bigram').T.to_dict('records')
# Create network plot 
G = nx.Graph()

# Create connections between nodes
for k, v in d[0].items():
    G.add_edge(k[0], k[1], weight=(v * 10))

fig, ax = plt.subplots(figsize=(14, 12))

pos = nx.spring_layout(G, k=1)

# Plot networks
nx.draw_networkx(G, pos,
                 font_size=16,
                 width=3,
                 edge_color='grey',
                 node_color='purple',
                 with_labels = False,
                 ax=ax)

# Create offset labels
for key, value in pos.items():
    x, y = value[0]+.135, value[1]+.045
    ax.text(x, y,
            s=key,
            bbox=dict(facecolor='red', alpha=0.25),
            horizontalalignment='center', fontsize=13)
    
plt.show()


# In[166]:


train = pd.read_csv('D:/ptnt/Suicide Detection/depressive_unigram_tweets_final.csv')


# In[167]:


train.head()


# In[168]:



train['word_count'] = train['tweet'].apply(lambda x: int(len(str(x).split(" "))))
train[['tweet','word_count']].head()


# In[170]:


train['char_count'] = train['tweet'].str.len() ## this also includes spaces
train[['tweet','char_count']].head()


# In[171]:


char_count = train['tweet'].str.len() ## includes spaces
char_count.head()


# In[172]:


char_count.mean()


# In[173]:


char_count.median()


# In[174]:


char_count.mode()


# In[175]:


word_counts = train['tweet'].apply(lambda x: int(len(str(x).split(" "))))


# In[177]:


word_counts.head()


# In[178]:


word_counts.mean()


# In[179]:


word_counts.median()


# In[180]:


word_counts.mode()


# In[181]:


# Plot Histogram on x
x = char_count
plt.hist(x, bins=50)
plt.gca().set(title='Characters', ylabel='Frequency', xlabel='characters');
plt.xlim(0, 300)


# In[182]:


x = word_counts
plt.hist(x, bins=50)
plt.gca().set(title='Words', ylabel='Frequency', xlabel='words');
plt.xlim(0, 100)


# In[ ]:




