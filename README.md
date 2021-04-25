# Online Shopping Behavior That Leads to Revenue
Project for STATS 503 (Multivariate Regression) at University of Michigan, where we were tasked to analyze a public dataset using statistical modeling techniques in R. 

# Project Goal
Determine the features most likely to lead to a sale from the Online Shoppers Purchasing Intention dataset from [UCI's ML archive](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset). This README summarizes the major steps and insights, but the finer details will be in [models.md](https://github.com/victle/online-shopper-behavior/blob/main/models.md). 

# Libraries 
Many of the modeling techniques are taken from the following packages:
* e1071 (for Naive Bayes and SVM)
* class (for KNN)

# Loading and Preprocessing Data
A lot of the columns have to be changed to represent a categorical variable.

![image](https://user-images.githubusercontent.com/26015263/115101381-ae273e80-9f11-11eb-84eb-5cd17b880f50.png)

It's important that, when doing a 70-30 split on the data, there is a balance in the the training and testing sets. 

# Exploratory Data Analysis 


# Modeling 
![image](https://user-images.githubusercontent.com/26015263/115101462-5937f800-9f12-11eb-9990-4badf606aafd.png)


## Classification Models Used
* Logistic Regression
  * Reduced Logistic Regression
* Naive Bayes
* LDA
* SVM
* Random Forest

## Table of Model Performance

|                     | Training Error | Testing Error |
|---------------------|---------------:|--------------:|
| Naive Bayes         |      0.1887601 |     0.1837838 |
| Logistic Regression |      0.1165701 |     0.1132739 |
| KNN                 |      0.1093859 |     0.1091892 |
| SVM                 |      0.0936269 |     0.1027027 |


