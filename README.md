# Stock-Price-Prediction-Challange

This code is part of the stock price prediction challenge from "Learn Python for Data Science #2" by Siraj Raval. 
The code works as follows: 
1. The User is asked to choose a stock ticker and a look_back period. 
2. Tweets about the stock and their sentiment are anlysed with Tweepy and TextBlob
3. If the overall sentiment is positive, stock data is downloaded from Yahoo-Finance
4. A NN is trained to predict t+1s stock price based on the look_back perdiod. 
5. The data and the predicted data is plotted. 

# Credits
Credits go to both Siraj Raval and Jason Brownlee for [this blog post](https://machinelearningmastery.com/time-series-prediction-with-deep-learning-in-python-with-keras/) from which I adapted code.  
