Online Shopping Dataset
================
Victor Le

``` r
#load data
library(readr)
online = read_csv('data.csv')
```

# Data Setup

``` r
#factorize categorical
online$OperatingSystems = factor(online$OperatingSystems)
online$Browser = factor(online$Browser)
online$Region = factor(online$Region)
online$TrafficType = factor(online$TrafficType)
online$VisitorType = factor(online$VisitorType)
online$Weekend = factor(online$Weekend)
online$Month = factor(online$Month)
online$Revenue = factor(online$Revenue)

#split 70-30
#want even amounts of YES/NO for Revenue
revenue_yes = which(online$Revenue == 'TRUE')
revenue_no = which(online$Revenue == 'FALSE')
set.seed(630)
# set.seed(1)

train_id = c(sample(revenue_yes, size = trunc(0.7 * length(revenue_yes))), 
             sample(revenue_no, size = trunc(0.7 * length(revenue_no))))
training = online[train_id, ]
testing = online[-train_id, ]

summary(training)
```

    ##  Administrative   Administrative_Duration Informational    
    ##  Min.   : 0.000   Min.   :   0.00         Min.   : 0.0000  
    ##  1st Qu.: 0.000   1st Qu.:   0.00         1st Qu.: 0.0000  
    ##  Median : 1.000   Median :   7.00         Median : 0.0000  
    ##  Mean   : 2.305   Mean   :  79.39         Mean   : 0.5071  
    ##  3rd Qu.: 4.000   3rd Qu.:  91.83         3rd Qu.: 0.0000  
    ##  Max.   :27.000   Max.   :3398.75         Max.   :16.0000  
    ##                                                            
    ##  Informational_Duration ProductRelated   ProductRelated_Duration
    ##  Min.   :   0.00        Min.   :  0.00   Min.   :    0.0        
    ##  1st Qu.:   0.00        1st Qu.:  7.00   1st Qu.:  182.2        
    ##  Median :   0.00        Median : 18.00   Median :  598.8        
    ##  Mean   :  34.03        Mean   : 31.91   Mean   : 1194.3        
    ##  3rd Qu.:   0.00        3rd Qu.: 38.00   3rd Qu.: 1458.3        
    ##  Max.   :2549.38        Max.   :686.00   Max.   :63973.5        
    ##                                                                 
    ##   BounceRates         ExitRates         PageValues        SpecialDay     
    ##  Min.   :0.000000   Min.   :0.00000   Min.   :  0.000   Min.   :0.00000  
    ##  1st Qu.:0.000000   1st Qu.:0.01429   1st Qu.:  0.000   1st Qu.:0.00000  
    ##  Median :0.003161   Median :0.02514   Median :  0.000   Median :0.00000  
    ##  Mean   :0.022368   Mean   :0.04339   Mean   :  5.921   Mean   :0.06155  
    ##  3rd Qu.:0.016798   3rd Qu.:0.05000   3rd Qu.:  0.000   3rd Qu.:0.00000  
    ##  Max.   :0.200000   Max.   :0.20000   Max.   :361.764   Max.   :1.00000  
    ##                                                                          
    ##      Month      OperatingSystems    Browser         Region      TrafficType  
    ##  May    :2360   2      :4633     2      :5585   1      :3384   2      :2729  
    ##  Nov    :2072   3      :1815     1      :1702   3      :1658   1      :1747  
    ##  Mar    :1367   1      :1763     4      : 527   2      : 807   3      :1454  
    ##  Dec    :1207   4      : 341     5      : 324   4      : 801   4      : 736  
    ##  Oct    : 386   8      :  56     6      : 121   6      : 578   13     : 526  
    ##  Aug    : 317   6      :  14     10     : 118   7      : 545   10     : 310  
    ##  (Other): 921   (Other):   8     (Other): 253   (Other): 857   (Other):1128  
    ##             VisitorType    Weekend      Revenue    
    ##  New_Visitor      :1164   FALSE:6621   FALSE:7295  
    ##  Other            :  60   TRUE :2009   TRUE :1335  
    ##  Returning_Visitor:7406                            
    ##                                                    
    ##                                                    
    ##                                                    
    ## 

# Standardizing the Dataset

``` r
#Check side-by-side boxplots for numerical variables
library(ggplot2)
library(dplyr)
library(tidyr)

# Standardize the numerical data in the training set and testing set using the mean and std of the training data set  
num_col = c('Administrative','Administrative_Duration', 'Informational','Informational_Duration','ProductRelated','ProductRelated_Duration', 'BounceRates','ExitRates','PageValues','SpecialDay')
training.norm = training
testing.norm = testing
for(i in num_col){
  mean_col = mean(pull(training,i))
  std_col  = sd(pull(training,i))
  training.norm[i] = scale(training[i], center = mean_col, scale = std_col)
  testing.norm[i]  = scale(testing[i], center = mean_col, scale = std_col)
}
```

# Naive Bayes

We want to start with some simple models and contextualize the results
that get. Let’s start with Naive Bayes, a tried and true first-starter.

``` r
# load the necessary packages for naive bayes
# install.packages('e1071')
library(e1071)
```

    ## Warning: package 'e1071' was built under R version 3.6.3

``` r
# Apply a Naive-Bayes to the model 
NB_model = naiveBayes(Revenue~., data = training.norm)
# print(NB_model)

# predictions
pred_train = predict(NB_model, newdata = training.norm)
mean(pred_train != training.norm$Revenue)
```

    ## [1] 0.1887601

``` r
pred_test  = predict(NB_model, newdata = testing.norm)
mean(pred_test != testing.norm$Revenue)
```

    ## [1] 0.1837838

``` r
table(pred_test, testing.norm$Revenue)
```

    ##          
    ## pred_test FALSE TRUE
    ##     FALSE  2611  164
    ##     TRUE    516  409

# Logistic Regression

Another simple, flexible model that we can use for predicting.

``` r
# Apply Logistic Regression to model 
logreg_model = glm(Revenue~., data = training.norm, family = binomial)
# summary(logreg_model)

# make predictions on the training set 
pred_train_probs = predict(logreg_model, training.norm)
```

    ## Warning in predict.lm(object, newdata, se.fit, scale = 1, type = if (type == :
    ## prediction from a rank-deficient fit may be misleading

``` r
# To get actually get predictions of the model, we need to convert back to probabilities
pred_train = binomial()$linkinv(pred_train_probs)
pred_train_log = rep('FALSE', nrow(training.norm))
pred_train_log[pred_train > 0.5] = 'TRUE'
mean(pred_train_log != training.norm$Revenue)
```

    ## [1] 0.1165701

``` r
# testing error (training data didn't contain TrafficType 17, so model couldn't train on it, so have to exclude it)
pred_test_probs = predict(logreg_model, testing.norm[testing.norm$TrafficType != 17,] )
```

    ## Warning in predict.lm(object, newdata, se.fit, scale = 1, type = if (type == :
    ## prediction from a rank-deficient fit may be misleading

``` r
pred_test = binomial()$linkinv(pred_test_probs)
pred_test_log = rep('FALSE', nrow(testing.norm[testing.norm$TrafficType != 17,]))
pred_test_log[pred_test > 0.5] = 'TRUE'
mean(pred_test_log != testing.norm[testing.norm$TrafficType != 17,]$Revenue)
```

    ## [1] 0.1132739

Turns out the testing error was reasonable given the training error.
Let’s see if we can improve upon the logistic regression model by only
fitting on significant predictors.

# Reduced Logistic Regression Model

``` r
# Using only significant predictors from a glm
logreg_model.red = glm(Revenue~ProductRelated_Duration + ExitRates + PageValues + Month + Browser+ TrafficType + VisitorType, data = training.norm, family = binomial)
# summary(logreg_model.red)

# training error
pred_train_probs = predict(logreg_model.red, training.norm)
pred_train = binomial()$linkinv(pred_train_probs)
pred_train_log = rep('FALSE', nrow(training.norm))
pred_train_log[pred_train > 0.5] = 'TRUE'
mean(pred_train_log != training.norm$Revenue)
```

    ## [1] 0.1158749

Ok, so the training error was only marginally better. Let’s see testing
error.

``` r
# testing error (training data didn't contain TrafficType 17, so model couldn't train on it, so have to exclude it)
pred_test_probs = predict(logreg_model.red, testing.norm[testing.norm$TrafficType != 17,] )
pred_test = binomial()$linkinv(pred_test_probs)
pred_test_log = rep('FALSE', nrow(testing.norm[testing.norm$TrafficType != 17,]))
pred_test_log[pred_test > 0.5] = 'TRUE'
mean(pred_test_log != testing.norm[testing.norm$TrafficType != 17,]$Revenue)
```

    ## [1] 0.1140849

Unfortunately, even with a reduced model, the testing and training
errors remain similar.

# KNN

Let’s see how a simple KNN would do.

``` r
# K-nearest neighbors 
library(class)
pred_train = knn(training.norm[num_col], training.norm[num_col], training$Revenue, k = 50)
table(pred_train, training$Revenue)
```

    ##           
    ## pred_train FALSE TRUE
    ##      FALSE  7101  749
    ##      TRUE    194  586

``` r
mean(pred_train != training$Revenue)
```

    ## [1] 0.10927

``` r
pred_test = knn(training.norm[num_col], testing.norm[num_col], training$Revenue, k = 50)
table(pred_test, testing$Revenue)
```

    ##          
    ## pred_test FALSE TRUE
    ##     FALSE  3047  322
    ##     TRUE     80  251

``` r
mean(pred_test != testing$Revenue)
```

    ## [1] 0.1086486

With KNN, we do have to do some cross-validation. The following section
simply validates our previous choice of k = 50.

### Cross-validation for value of K

``` r
k_vec = c(1,5,11,51,101,201,401)

errors_train = c()
errors_test  = c()
for(j in 1:length(k_vec)){
  
  pred_train =  knn(training.norm[num_col], training.norm[num_col], training$Revenue, k = k_vec[j])
  
  pred_test = knn(training.norm[num_col], testing.norm[num_col], training$Revenue, k = k_vec[j])
  
  errors_test[j] = mean(pred_test!= testing$Revenue)
  errors_train[j] = mean(pred_train != training$Revenue)
}

errors = data.frame(errors_train, errors_test, k_vec)

ggplot(errors, aes(x=k_vec)) + geom_line(aes(y = errors_train), col = 'red') + geom_point(aes(y=errors_train), col = 'red') +  geom_line(aes(y=errors_test), col = 'blue') + geom_point(aes(y=errors_test),col='blue')
```

![](models_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

# SVM

``` r
# Apply SVM to model 

library(e1071)

# with SVM, we can simply use tune to determine the best hyperparameters
# tune.out = tune(svm, Revenue~., data = training.norm, ranges = list(cost = c(0.1, 1, 10, 100), kernel = c('linear','radial')))
# summary(tune.out)
```

From the output, it seems like the best hyperparameters were C=10, and
the radial kernel.

``` r
# final model
svm_model = svm(Revenue~., data = training.norm, cost = 10, kernel = 'radial')
# summary(svm_model)
pred_train = predict(svm_model, newdata = training.norm)
mean(pred_train != training.norm$Revenue)
```

    ## [1] 0.09362688

``` r
pred_test  = predict(svm_model, newdata = testing.norm)
mean(pred_test != testing.norm$Revenue)
```

    ## [1] 0.1027027

``` r
# Confusion matrix
table(pred_test, testing.norm$Revenue)
```

    ##          
    ## pred_test FALSE TRUE
    ##     FALSE  3030  283
    ##     TRUE     97  290

That’s a good set of models! Let’s organize our results and see where we
stand now.

# Final Results

``` r
library(knitr)
```

    ## Warning: package 'knitr' was built under R version 3.6.3

``` r
model_errors.1 = c(0.1887601, 0.1837838)
model_errors.2 = c(0.1165701, 0.1132739)
model_errors.3 = c(0.1093859, 0.1091892)
model_errors.4 = c(0.09362688, 0.1027027)
model_errors = rbind(model_errors.1, model_errors.2, model_errors.3, model_errors.4)
colnames(model_errors) = c('Training Error', 'Testing Error')
rownames(model_errors) = c('Naive Bayes', 'Logistic Regression','KNN','SVM')
knitr::kable(model_errors)
```

|                     | Training Error | Testing Error |
|---------------------|---------------:|--------------:|
| Naive Bayes         |      0.1887601 |     0.1837838 |
| Logistic Regression |      0.1165701 |     0.1132739 |
| KNN                 |      0.1093859 |     0.1091892 |
| SVM                 |      0.0936269 |     0.1027027 |
