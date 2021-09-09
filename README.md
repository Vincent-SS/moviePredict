# moviePredict
Introduction

##In this assignment you will be using the Movie dataset provided and the machine learning algorithm you have learned in this course in order to find out: knowing only things you could know before a film was released , what the rating and revenue of the film would be. the rationale here is that your client is a movie theater that would like to decide how long should they reserve the movie theater for to show a movie when it is released.

##Datasets

In this assignment, you will be given two datasets [training.csv](https://github.com/mysilver/COMP9321-Data-Services/raw/master/20t1/assign3/training.csv) and [validation.csv](https://github.com/mysilver/COMP9321-Data-Services/raw/master/20t1/assign3/validation.csv).

You can use the training dataset (but not validation) for training machine learning models, and you can use validation dataset to evaluate your solutions and avoid over-fitting.

##Part-I: Regression
In the first part, you are asked to predict the "revenue" of movies based on the information in the provided dataset. More specifically, you need to predict the revenue of a movie based on a subset (or all) of the following attributes (**make sure you DO NOT use rating** ):

cast,crew,budget,genres,homepage,keywords,original_language,original_title,overview,production_companies,production_countries,release_date,runtime,spoken_languages,status,tagline

##Part-II: Classification
Using the same datasets, you must predict the rating of a movie based on a subset (or all) of the following attributes (**make sure you DO NOT use revenue** ):

cast,crew,budget,genres,homepage,keywords,original_language,original_title,overview,production_companies, production_countries,release_date,runtime,spoken_languages,status,tagline
