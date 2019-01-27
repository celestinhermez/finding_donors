# Finding Donors for Charity ML

## Introduction

This project shows how we can use supervised learning to solve a business problem. Charity ML is 
an imaginary non-profit which survives from donations. In particular, past experience has shown
that people making more than $50,000 a year were much more likely to donate. As a result, our goal
is to use publicly available data (from the Census information) to build a model to predict someone's
income, thus allowing Charity ML to focus its outreach efforts to people most likely to donate.

 The dataset for this project originates from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Census+Income). 
 The datset was donated by Ron Kohavi and Barry Becker, after being published in the article 
 _"Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid"_. 
 We can find the article by Ron Kohavi [online](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf). 
 The data we investigate here consists of small changes to the original dataset, 
 such as removing the `'fnlwgt'` feature and records with missing or ill-formatted entries.
 
 Our approach will consist in building several models on data from the 1994 US census. We will then
 select the best candidate algorithm and tune it further. We use a supervised learning approach since
 we have a target variable (binary, whether an individual makes more than $50,000) which we want to model.
 
 ## Repository Description
 
 In order to run this project, you can simply clone this repository. In particular, `visuals.py`
 contains helper functions to help us visualize our data and the model outputs. The census data
 is included in `census.csv`.
 
 The analysis is contained in `finding_donors.ipynb`, which an HTML version in `finding donors.html`.
 In order to run the notebook, you will need the following libraries:
 * numpy
 * pandas
 * Ipython.display
 * time
 * sklearn
 
 ## Analysis
 
 We built a robust model _CharityML_ can use to identify potential donors.
 
 ### Preprocessing
 
 Given its source, our dataset is already clean, but we do not to apply some preprocessing:
 * transform skewed features by applying a logarithmic transformation
 * normalize features through a `MinMaxScaler`
 * encode categorical features through one-hot encoding
 * encode the categorical target variable by a numeric binary variable
 
 ### Modeling
 
 We build four models:
 * a naive predictor, that always predicts an individual makes more than \$50,000. This serves
 as a benchmark the other model have to improve upon
 * Naive Bayes
 * Adaboost
 * Support Vector Machines
 
 In order to choose between these three models, we rely both on accuracy and F<sub>0.5</sub> score 
 to have a good balance between precision and recall (placing a bit more emphasis on 
 precision). Based on these criteria, AdaBoost combines high performance with fast training. We
 further tune its hyperparameters through grid search:
 * number of estimators (`n_estimators`): 10, 25, 50
 * learning rate (`learning_rate`): 0.5, 1, 2
 * maximum depth of the individual decision trees (`base_estimator__max_depth`): 1, 5, 10
 
 After tuning, we see improvement in performance, both for the F<sub>0.5</sub> score and the accuracy compared
 to the original model. It is also an approximately 3x increase compared to our naive predictor, which
 was our original benchmark.
 
 ### Feature Selection
 
 The 5 most important features in the model are `capital_loss`, `age`, `capital_gain`, `hours_per_week`
 and `education_num`. With that in mind, we can fit a model with only these most important features, at very
 little cost in accuracy and F<sub>0.5</sub> score. As a result, if speed is of the essence for the training
 of this model, I would recommend focusing on these features.
 
 ## Acknowledgments
 
 This project was part of Udacity's _Data Scientist_ nanodegree. They provided the data, as well
 as the original instructions for this project. All of the code is my own unless specified otherwise.
 
 
