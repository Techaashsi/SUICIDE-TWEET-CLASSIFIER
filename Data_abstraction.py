import tweepy
from tweepy import OAuthHandler
import csv
import time

#API Credentials
consumer_key = 'EzrugiA86Zy1HMzUpw1PAGeMV'
consumer_secret = 'r0ONLHd9MgKGH1BRIoRX1PBeOHTNY6LRGCWb777uP0166nDYLr'
access_token = '1253686212818350087-I1qXWLcdLKDwuFqJ05m3hOqa8Gv6Ah'
access_secret = 'dqbCkTOQXRDXqiw0XKTKHY4xJjZQ5r5ZRNAyZEdii7Qig'

auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_key, access_secret)

#Verification of API
api = tweepy.API(auth,wait_on_rate_limit=True)

#Print Twiiter Account Name to Verify Credentials
print(api.me().name)

searchTerm = input("Enter Keyword/Tag to search about: ")
NoOfTerms = int(input("Enter how many tweets to search: ")

tweets = tweepy.Cursor(api.search, q=searchTerm, lang = "en",since="2021-02-18").items(NoOfTerms) 
'''
Change the date from which you would like to take'''

data = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])
data.head(10)

def Clean_Text(text): 
        return ' '.join(re.sub("(RT|@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(U 30FC)", " ", text.strip()).split())
        
data['Tweets']=data['Tweets'].apply(Clean_Text)
data_copy=data.drop_duplicates(['Tweets'])
data_copy=data.reset_index(drop=True)
data_copy

data_copy['Analysis']=data_copy['Polarity'].apply(get_analysis)


df.to_excel (r'file_name', index = False, header=True)
