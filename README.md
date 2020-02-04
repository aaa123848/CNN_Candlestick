# CNN_for_Candlestick

## cnn for candle stick category

this project use convolution network training a model to find pattern in candlestick and do the classification

the pattern we detect including evening star, morning star, bearish engulfing, bullish engulfing,
shooting star, inverted hammer, bearish harami, bullish harami

the val_accuracy is above 80% 


## requirement

keras = 2.2.5 

tensorflow-gpu = 1.10.0 

numpy = 1.17.0

## process

find the data by this url: https://drive.google.com/file/d/1GKPTTBF-Kl8Wu6LHlBpEbqURbD13RNgc/view?usp=sharing

use predict.py to detect the result with test data 

and see our model structure in keras_cnn.py

use show_result_picture to see the picture with result




## result

the result with test data :

| Class |  accuracy(%)  |
|----------|-------------|
| No class | 73% | 
| Evening star | 97% | 
| Morning star | 90% | 
| Bearish engulfing | 84% | 
| Bullish engulfing | 80% | 
| Shooting star | 91%| 
| Inverted hammer | 91% | 
| Bearish harami | 81% | 
| Bullish harami | 87% | 


## reference
Encoding Candlesticks as Images for Patterns Classification Using Convolutional Neural Networks
--Yun-Cheng Tsai, Jun Hao Chen, Chun-Chieh Wang
