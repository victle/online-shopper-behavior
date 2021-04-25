# Online Shopping Behavior That Leads to Revenue
Project for STATS 503 (Multivariate Regression) at University of Michigan, where we were tasked to analyze a public dataset using statistical modeling techniques in R. 

# Project Goal
Determine the features most likely to lead to a sale from the Online Shoppers Purchasing Intention dataset from [UCI's ML archive](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset). This README summarizes the major steps and insights, but the finer details will be in [models.md](https://github.com/victle/online-shopper-behavior/blob/main/models.md). An Rpubs version can be found [here](https://rpubs.com/victle/onlineShoppingEDA).

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
## KNN Cross-validation
![image](https://user-images.githubusercontent.com/26015263/115101462-5937f800-9f12-11eb-9990-4badf606aafd.png)

## Decision Trees
![unnamed-chunk-5-1](https://user-images.githubusercontent.com/26015263/115981036-f58b7b80-a55e-11eb-831c-0ce325c8fbf0.png)

## Feature Importance
![unnamed-chunk-6-1](https://user-images.githubusercontent.com/26015263/115981052-105df000-a55f-11eb-84b9-7ff6440a4df1.png)


## Classification Models Used
* Logistic Regression
  * Reduced Logistic Regression
* Naive Bayes
* LDA
* SVM
* Random Forest
* Adaboost

## Table of Model Performance

|                     | Training Error | Testing Error |
|---------------------|---------------:|--------------:|
| Naive Bayes         |      0.1887601 |     0.1837838 |
| Logistic Regression |      0.1165701 |     0.1132739 |
| KNN                 |      0.1093859 |     0.1091892 |
| SVM                 |      0.0936269 |     0.1027027 |
| Random Forest       |      0.0000000 |     0.0918919 |
| Adaboost            |      0.0085747 |     0.1035135 |

# Comments on PageValue as a feature
PageValues is an incredibly important feature in this dataset for predicting revenue. The biggest problem here is that PageValues is computed by dividing the revenue of a shopping instance by the number of views a page gets. This inherently encodes revenue into the problem, and does not really reflect the volitional behavior of the shopper. In fact, PageValue by itself does a great job at classifying Revenue on its own (achieving about 10% error on the testing set).
