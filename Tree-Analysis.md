Online Shopping - Tree Analysis
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

We will start with an unconstrained tree, just to get a base line for
the training and testing errors.

``` r
#Unconstrained Tree
library(rpart)
library(rpart.plot)  
library(rattle)
```

    ## Loading required package: tibble

    ## Loading required package: bitops

    ## Rattle: A free graphical interface for data science with R.
    ## Version 5.4.0 Copyright (c) 2006-2020 Togaware Pty Ltd.
    ## Type 'rattle()' to shake, rattle, and roll your data.

``` r
# Fit a large tree (cp = 0) using the Gini index 
tree1 = rpart(Revenue ~ ., training, parms = list(split = "gini"), method = "class", cp = 0)

# We can plot it if we want, but it's honestly unintelligible
# fancyRpartPlot(tree1)

# Errors
tree1_pred_training = predict(tree1, newdata =  training, type = "class")
tree1_training_err = mean(tree1_pred_training != training$Revenue)
tree1_pred_testing = predict(tree1, newdata =  testing, type = "class")
tree1_testing_err = mean(tree1_pred_testing != testing$Revenue)
cat("Training Error: ", tree1_training_err, "\nTesting Error: ", tree1_testing_err)
```

    ## Training Error:  0.0631518 
    ## Testing Error:  0.1135135

Honestly, 11% testing errors isn’t all that bad compared to some of the
other models we tried.

# Cross-validation to tune **cp**

Now to improve the model, we will use cross validation to tune the
hyperparameter parameter **cp** (cost-complexity pruning), and prune our
tree to generate a smaller model.

``` r
#Find best cp using a simple plotcp function
tree.cp = rpart(Revenue ~ ., data = training, parms = list(split = "gini"), method = "class")
plotcp(tree.cp)
```
![unnamed-chunk-4-1](https://user-images.githubusercontent.com/26015263/115981002-c07f2900-a55e-11eb-867f-50b5533b2621.png)
A rule of thumb here is that you should probably choose the value of leftmost
value of **cp** where your relative error + the standard deviation falls
below the cross-validation error. In our case, this would **cp** =
0.012.

``` r
# New tree using the best cp
tree.cp2 = rpart(Revenue ~ ., training, parms = list(split = "gini"), method = "class", cp = 0.012)
fancyRpartPlot(tree.cp2, cex = 0.7)
```

![unnamed-chunk-5-1](https://user-images.githubusercontent.com/26015263/115981006-c96ffa80-a55e-11eb-80d2-80c0e7962cf6.png)


``` r
# Errors
tree2_pred_training = predict(tree.cp2, newdata =  training, type = "class")
tree2_training_err = mean(tree2_pred_training != training$Revenue)
tree2_pred_testing = predict(tree.cp2, newdata =  testing, type = "class")
tree2_testing_err = mean(tree2_pred_testing != testing$Revenue)
cat("Training Error: ", tree2_training_err, "\nTesting Error: ", tree2_testing_err)
```

    ## Training Error:  0.1 
    ## Testing Error:  0.1032432

We can see a slight improvement (but it’s not much) in the testing error
with this simpler model.

# Variable Importance of Single Tree Model

Although it clearly seems like “PageValues” is most important feature,
let’s take a formal look with a variable importance plot.

``` r
#importance plot
library(vip)
```

    ## 
    ## Attaching package: 'vip'

    ## The following object is masked from 'package:utils':
    ## 
    ##     vi

``` r
vip(tree.cp2, geom = "col")
```

![unnamed-chunk-6-1](https://user-images.githubusercontent.com/26015263/115981008-cd038180-a55e-11eb-80ac-08df34c06072.png)

# Random Forests

Now of course, in lieu of having nice interpretability with the single
tree, we can create a random forest that will reduce the variance error
in our models. We can start off with using some of the default values.

``` r
library(randomForest)
```

    ## randomForest 4.6-14

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:rattle':
    ## 
    ##     importance

``` r
rf_Revenue = randomForest(Revenue ~ ., data = training, mtry = floor(sqrt(17)), importance = TRUE)
rf_Revenue
```

    ## 
    ## Call:
    ##  randomForest(formula = Revenue ~ ., data = training, mtry = floor(sqrt(17)),      importance = TRUE) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 4
    ## 
    ##         OOB estimate of  error rate: 9.58%
    ## Confusion matrix:
    ##       FALSE TRUE class.error
    ## FALSE  7007  288   0.0394791
    ## TRUE    539  796   0.4037453

``` r
importance(rf_Revenue)
```

    ##                              FALSE         TRUE MeanDecreaseAccuracy
    ## Administrative          26.3515718  -6.13249187           25.3930721
    ## Administrative_Duration 24.5667357  -7.38400558           22.4092873
    ## Informational           13.0502862   0.02036153           12.3593117
    ## Informational_Duration  16.7200418   0.55021355           16.4602580
    ## ProductRelated          29.6036437   3.24335040           32.8521677
    ## ProductRelated_Duration 25.3117641   8.40371701           30.8602106
    ## BounceRates             17.3437032  23.07487838           28.2272182
    ## ExitRates               17.5058807  32.00787140           30.9140309
    ## PageValues              86.4689236 220.17231197          136.0073153
    ## SpecialDay              -3.7523178   4.74079689           -0.3938124
    ## Month                    1.4451065  40.45663667           22.7165821
    ## OperatingSystems         8.3123030   1.18537267            8.4429770
    ## Browser                 13.1566151  -4.00741214           10.4193896
    ## Region                  -0.4845472   1.65779272            0.5335825
    ## TrafficType             11.5285433   6.49362658           13.4093875
    ## VisitorType             17.4584763   7.81826063           19.4382907
    ## Weekend                 -1.1530810   4.45679124            1.4975691
    ##                         MeanDecreaseGini
    ## Administrative                 86.980118
    ## Administrative_Duration       118.399577
    ## Informational                  36.222446
    ## Informational_Duration         53.841231
    ## ProductRelated                156.553459
    ## ProductRelated_Duration       185.956965
    ## BounceRates                   115.328399
    ## ExitRates                     182.100725
    ## PageValues                    818.621328
    ## SpecialDay                      7.031764
    ## Month                         129.524663
    ## OperatingSystems               40.856872
    ## Browser                        53.617655
    ## Region                        104.292300
    ## TrafficType                   109.734819
    ## VisitorType                    24.162413
    ## Weekend                        18.458081

``` r
varImpPlot(rf_Revenue)
```

![unnamed-chunk-7-1](https://user-images.githubusercontent.com/26015263/115981015-d1c83580-a55e-11eb-8fd3-8ee0504f27a5.png)


``` r
# Errors
rf_Revenue_pred_training = predict(rf_Revenue, newdata =  training, type = "class")
rf_Revenue_training_err = mean(rf_Revenue_pred_training != training$Revenue)
rf_Revenue_pred_testing = predict(rf_Revenue, newdata =  testing, type = "class")
rf_Revenue_testing_err = mean(rf_Revenue_pred_testing != testing$Revenue)
cat("Training Error:", rf_Revenue_training_err, "\nTesting Error: ", rf_Revenue_testing_err)
```

    ## Training Error: 0 
    ## Testing Error:  0.09486486

Using a Random Forest approach, we can obtain a slight decrease in the
testing error. The number of features to try for each tree was a rule of
thumb, were we used the square root of the total number of features.
However, let’s see if we can tune the number of trees to improve
performance.

## Tuning the **ntree** hyperparameter

``` r
i = 1
train_errors_vec = c()
test_errors_vec = c()
ntree_vec = seq(600,2000,200)
for (ntree in ntree_vec) {
  # fit a tree using current number of ntree
  set.seed(630)
  rf.ntree = randomForest(Revenue ~ ., data = training, mtry = floor(sqrt(17)), ntree = ntree, importance = TRUE)
  
  # grab training and testing errors 
  currTrainPred = predict(rf.ntree, newdata = training, type = "class")
  currTrainError = mean(currTrainPred != training$Revenue)
  currTestPred = predict(rf.ntree, newdata = testing, type = "class")
  currTestError = mean(currTestPred != testing$Revenue)
  
  # add to the lists above 
  train_errors_vec[i] = currTrainError
  test_errors_vec[i] = currTestError
  i = i + 1
}
# combine into a data frame
ntree_df = data.frame(train_errors_vec, test_errors_vec, ntree_vec)
```

After plotting the testing errors as a function of **ntrees**, there
isn’t really a definitive pattern or nice elbow. For now, we’ll stick
with 1000 **ntrees** as it gives us the lowest testing error. Though,
the amount of change is quite minimal.

``` r
plot(ntree_vec, ntree_df$test_errors_vec, ylab = "Testing Error", xlab = "Number of trees")
```

![unnamed-chunk-9-1](https://user-images.githubusercontent.com/26015263/115981019-d5f45300-a55e-11eb-8a9c-af0363e9c88d.png)


``` r
# fit the final model
rf.ntree = randomForest(Revenue ~ ., data = training, mtry = floor(sqrt(17)), ntree = 1000, importance = TRUE)

# grab training and testing errors
currTrainPred = predict(rf.ntree, newdata = training, type = "class")
currTrainError = mean(currTrainPred != training$Revenue)
currTestPred = predict(rf.ntree, newdata = testing, type = "class")
currTestError = mean(currTestPred != testing$Revenue)
cat("Training Error: ", currTrainError, "\nTesting Error: ", currTestError)
```

    ## Training Error:  0 
    ## Testing Error:  0.09189189

# Boosting

We can try one more technique to improve the performance of our working
model, and that is boosting, whereby trees are fitted to reweighted
versions of the training data. And ultimately, the data is classified
using a weighted majority vote.

``` r
library(gbm)
```

    ## Loaded gbm 2.1.8

``` r
# We actually have to do some pre-processing and set the Revenue variable to be 0 or 1.
training$Revenue = ifelse(training$Revenue == TRUE, 1, 0)
testing$Revenue = ifelse(testing$Revenue == TRUE, 1, 0)
ada_shopping = gbm(Revenue ~ ., data = training, distribution = "adaboost", n.trees = 5000, interaction.depth = 3)

# we can even play with shrinkage parameters to determine training speed if we wanted to
# ada_shopping = gbm(Revenue ~ ., data = training, distribution = "adaboost", n.trees = 10000, interaction.depth = 3, shrinkage = 0.01)
summary(ada_shopping)
```

![unnamed-chunk-10-1](https://user-images.githubusercontent.com/26015263/115981022-d8ef4380-a55e-11eb-8ccf-a7d646e93441.png)


    ##                                             var     rel.inf
    ## PageValues                           PageValues 32.87144600
    ## Month                                     Month 12.49426029
    ## ProductRelated_Duration ProductRelated_Duration  9.52882663
    ## TrafficType                         TrafficType  8.41372313
    ## ProductRelated                   ProductRelated  5.98441563
    ## ExitRates                             ExitRates  5.61249906
    ## Region                                   Region  5.07447872
    ## Administrative_Duration Administrative_Duration  4.90778705
    ## BounceRates                         BounceRates  4.67980761
    ## Browser                                 Browser  2.46517088
    ## Informational_Duration   Informational_Duration  2.37605270
    ## Administrative                   Administrative  2.10774025
    ## OperatingSystems               OperatingSystems  1.39381660
    ## Informational                     Informational  0.98627416
    ## VisitorType                         VisitorType  0.71244837
    ## Weekend                                 Weekend  0.29820183
    ## SpecialDay                           SpecialDay  0.09305111

``` r
# compute testing errors
ada_predict = predict(ada_shopping, newdata = testing, type = "response")
```

    ## Using 5000 trees...

``` r
ada_predict = ifelse(ada_predict > 0.5, 1, 0)
ada_error = mean(ada_predict != testing$Revenue)
sprintf("Testing Error: %s", ada_error)
```

    ## [1] "Testing Error: 0.104054054054054"

Unfortunately, it seems like the testing errors aren’t improving much,
if at all with ensemble techniques. Even then, any changes are likely to
be minimal at best. At this point, it seems like re-assessing the
problem is required to substantially improve predictive performance,
such as collecting more data.

# Big Assumptions with PageValues

However, something doesn’t sit right with me. PageValues is an
incredibly important feature in this dataset for predicting revenue. The
biggest problem here is that PageValues is computed by dividing the
revenue of a shopping instance by the number of views a page gets. This
inherently encodes revenue into the problem, and does not really reflect
the volitional behavior of the shopper. In fact, PageValue by itself
does a great job at classifying Revenue on its own (achieving about 10%
error on the testing set).

``` r
tree.pv = rpart(Revenue ~ PageValues, data = training, parms = list(split = "gini"), method = "class", cp = 0.012)

pv_pred_testing = predict(tree.pv, newdata =  testing, type = "class")
pv_testing_err = mean(pv_pred_testing != testing$Revenue)
sprintf("Testing Error: %s", pv_testing_err)
```

    ## [1] "Testing Error: 0.105945945945946"
