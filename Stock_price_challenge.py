# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 13:33:54 2019

@author: nilsh
"""

import tweepy
import csv
import numpy as np
from textblob import TextBlob
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd
import pdb

#Step 1 - Twitter Api keys
consumer_key = "8DJOSVDajdKJtOvkVkpZwbLnA"
consumer_secret = "WoIcFRZbEojAF5HRotF2y5HQTgdzrZnJI0zTWIdBmHnETHsAFZ"
access_token = "714807435559628800-o0JMhiqQTyY5JLAgly5I2ooXV0kesRk"
access_token_secret = "F1tm2oajhLaEB7pw2v7Bo0joIJfGXSsczZ2F00b9LGX7Y"
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Function to Retrieve the Twitter sentiment
def get_twitter_sentiment(TICKR,count=100):
    public_tweets=api.search(TICKR,count=count)
    sentiments=[]
    
    for tweet in public_tweets:
        print(tweet.text)
        analysis=TextBlob(tweet.text)
        sentiments.append(np.sign(analysis.sentiment.polarity))
        
        sentiment_inequality=np.sum(sentiments)
    
    if sentiment_inequality >= 0:
        print("The overall sentiment is positive")
        return True
    else:
        print("Unfortunately the overall sentiment is negative")
        return False
# Alternative 1 - loading stored datafile
dates = []
prices = []
def get_data(filename):
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)
		for row in csvFileReader:
			dates.append(int(row[0].split("-")[0]+row[0].split("-")[1]+row[0].split("-")[2]))
			prices.append(float(row[1]))
	return

# Alternative 2 - downloading Ticker data from yahoo-finance
def load_data1(TICKR,start="2017-01-01"):
    pd.core.common.is_list_like=pd.api.types.is_list_like
    import fix_yahoo_finance as fyf
    from pandas_datareader import data as pdr
    fyf.pdr_override()
    
    tickr=pdr.get_data_yahoo(TICKR,start=start)
    for i in range(len(tickr)):
        dates.append(int(str(tickr.index[i]).split("-")[0]+str(tickr.index[i]).split("-")[1]+
                             str(tickr.index[i]).split("-")[2].split(":")[0][:2]))
        prices.append(float(tickr["Adj Close"].iloc[i]))
    
    return dates,prices

# Loads in data from Yahoo-finance and returns the Adjusted price data in an np.array
def load_data(TICKR,start="2017-01-01"):
    pd.core.common.is_list_like=pd.api.types.is_list_like
    import fix_yahoo_finance as fyf
    from pandas_datareader import data as pdr
    fyf.pdr_override()
   
    tickr=pdr.get_data_yahoo(TICKR,start=start)
    tickr=tickr["Adj Close"].values
    tickr=tickr.reshape(-1,1)
    tickr=tickr.astype("Float32")
    
    return tickr

# Apply NN to price data with dates and prices
def predict_prices1(dates, prices,look_ba):
    y_train=np.reshape(prices,(len(prices),1))
    X_train=np.reshape(dates,(len(dates),1))

    model = Sequential()
    model.add(Dense(8,input_dim=1,activation="relu")) # hidden layer 1
    model.add(Dense(8,activation="relu")) # hidden layer 2
    model.add(Dense(1,init="uniform",activation="relu")) # output layer
    model.compile(optimizer="adam",loss="mean_squared_error") #metrics is list of metrics to be evaluated by model
    model.fit(X_train,y_train,epochs=1000) 

# Apply NN to price data with lagged price data
def predict_prices(TICKR,data,look_back):
    train_size=int(len(data)*0.67)

    train,test=data[0:train_size,:],data[train_size:len(data),:]
    
    TrainX,TrainY=create_dataset(train,look_back)
    TestX,TestY=create_dataset(test,look_back)
    
    model=Sequential()
    model.add(Dense(12,input_dim=look_back,activation="relu"))
    model.add(Dense(8,activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error",optimizer="adam")
    model.fit(TrainX,TrainY,epochs=400,batch_size=2,verbose=2)

    trainPredict=model.predict(TrainX)
    testPredict=model.predict(TestX)
    
    #Shift train predictions for plotting
    trainPredictPlot=np.empty_like(data)
    trainPredictPlot[:,:]=np.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back,:]=trainPredict
    
    #Shift test predictions for plotting
    testPredictPlot=np.empty_like(data)
    testPredictPlot[:,:]=np.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(data)-1,:]=testPredict

    #plot baseline and predicitons
    plt.plot(data,label="Original Data")
    plt.plot(trainPredictPlot,label="Fit Trainin Data")
    plt.plot(testPredictPlot, label="Fit Testing Data")
    plt.title("Chart of Ticker: {}".format(TICKR))
    plt.legend(loc=0)
    plt.show()

# Create dataset based on lagged price data
def create_dataset(data,look_back=1):
    dataX,dataY=[],[]
    for i in range(len(data)-look_back-1):
        a=data[i:(i+look_back),0]
        dataX.append(a)
        dataY.append(data[i+look_back,0])
    return np.array(dataX),np.array(dataY)

# Running the routine
def run_code():
    TICKR = input("Please enter a stock ticker to retrieve data from:").upper()
    look_back=input("Please enter a look_back period for the algorithm >= 1:")
    
    if get_twitter_sentiment(TICKR,count=100):
        #dates, prices = load_data(TICKR,start="2017-01-01")
        #predict_prices(dates, prices)
        data=load_data(TICKR)
        predict_prices(TICKR,data,int(look_back))
    else: 
        print("Please run the code again with a different Ticker!")
 
    return

run_code()





    





















