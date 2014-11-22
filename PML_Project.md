####Practical Machine Learning - Human Activity Recognition Project#####
######By Jean Rene Ndeki######
######Wednesday, November 19, 2014######

####I. Introduction####
The Human Activity Recognition (HAR) project consists of predicting participants' activities using exercise data. The potential applications of the project include elderly monitoring, life log systems for monitoring energy expenditure, weight-loss program support, and digital assistants for weight lifting exercises. Data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants are gathered and used for modeling. The question the project tries to answer is how to predict the manner in which the participants will perform their exercise. 

First, a working directory is set in a local machine drive to store the project data and files.

```r
dr<-'C:/Users/John/Desktop/GeekSquad Back-Up/Users/Jean-Rene/Desktop/Coursera/10. Practical Machine Learning/Peer Assessment Project'
setwd(dr)
```

Secondly, the libraries required to implement the project are initialized.

```r
library(AppliedPredictiveModeling)
library(caret)
library(kernlab)
library(randomForest)
```

Next, the following methodlogy is used: 1. Question definition (see above) 2. Data gathering, cleaning, and processing 3. Feature selection 4.Algorithm selection 5. Prediction 6. Evaluation 

####II. Data Source, Download, and Processing####
The data source is "http://groupware.les.inf.puc-rio.br/har". The files are downloaded from the listed urls to the local directory.

The training file is downloaded.

```r
URLtrain <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"

filetrain <- "C:/Users/John/Desktop/GeekSquad Back-Up/Users/Jean-Rene/Desktop/Coursera/10. Practical Machine Learning/Peer Assessment Project/pml-training.csv"

download.file(url=URLtrain, destfile=filetrain, method="curl")
```

The tidy training set blank or NA entries are flagged NA strings.

```r
training_t <- read.csv("pml-training.csv",row.names=1,na.strings= c("NA",""," "))
```

The test file is downloaded.

```r
URLtest <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

filetest <- "C:/Users/John/Desktop/GeekofSquad Back-Up/Users/Jean-Rene/Desktop/Coursera/10. Practical Machine Learning/Peer Assessment Project/pml-testing.csv"

download.file(url=URLtest, destfile=filetest, method="curl")
```

The tidy test set blank or NA entries are flagged NA strings.

```r
testing_t <- read.csv("pml-testing.csv",row.names=1,na.strings= c("NA",""," "))
```

Clean training and test sets are obtained. Columns with (N/A) or blank ("", " ") entries are removed. The resulting sets include all relevant variables, except the first seven fields.

```r
# clean training set
training_c<-training_t[,sapply(training_t, function(x) all(!is.na(x)))] 

# clean training set with relevant predictors (removal of the first six fields)
training_cm<-training_c[7:length(training_c)]
```


```r
# clean test set
testing_c<-testing_t[,sapply(testing_t, function(x) all(!is.na(x)))]

# clean test set with relevant predictors (removal of the first six fields)
testing_cm<-testing_c[7:length(testing_c)]   
```

The study design requires a cross validation set. The clean training set is partitioned into a training set (70%) and cross validation set (30%). Three files (training, cross validation, and testing) will be used for the study.


```r
inTrain <- createDataPartition(y = training_cm$classe, p = 0.7, list = FALSE)

training   <- training_cm[inTrain, ]        # training set

validation <- training_cm[-inTrain, ]       # cross validaton set

testing    <- testing_cm                    # testing set
```

####III. Features ####
The features are shown in appendix VIII.1 (Exploratory Data Analysis). Notice, the training and cross validation sets have similar features, 53 variables. The test set differs from the training and cross validation sets in two fields. The outcome "classe" exists in the training and cross validation sets. The  problem id is only in the test set.The study suggests that any of the other variables may be used to predict with.

####IV. Algorithm and Prediction####
The random forest algorithm is applied. Multiple deep decision trees, trained on different parts of the same training set, are averaged with the goal of reducing the variance. Bootstrap aggregating or bagging is applied: sampling with replacement, decision tree training, and predictions' average for the majority vote selection. The model is created using the training set. 

#####IV.1 Random Forest Model#####
The prediction model is defined as follows:

```r
modFit <- randomForest(classe ~ ., data = training)
```

The solution indicates the model, type of random forest (classification), number of trees (500), number of variables tried at each split (7),the OOB estimate of error rate (.52%), and the confusion matrix. This includes each class error.

```r
print(modFit)
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.53%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3902    2    0    1    1 0.001024066
## B   12 2641    5    0    0 0.006395786
## C    0   14 2378    4    0 0.007512521
## D    0    0   23 2227    2 0.011101243
## E    0    0    3    6 2516 0.003564356
```

####V. Cross Validation and Testing####

#####V.1 Cross Validation Prediction#####

The cross validation is performed using the model developed from the training set and applying it to the validation set.

```r
#Cross validation
predcv <- predict(modFit, validation)
```

The confusion matrix is produced below. Additional results figure in appendix VIII.2. 

```r
  #Confusion matrix
confusionMatrix(validation$classe, predcv)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    4 1134    1    0    0
##          C    0    7 1019    0    0
##          D    0    0    7  957    0
##          E    0    0    0    3 1079
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9963          
##                  95% CI : (0.9943, 0.9977)
##     No Information Rate : 0.2851          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9953          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9976   0.9939   0.9922   0.9969   1.0000
## Specificity            1.0000   0.9989   0.9986   0.9986   0.9994
## Pos Pred Value         1.0000   0.9956   0.9932   0.9927   0.9972
## Neg Pred Value         0.9991   0.9985   0.9984   0.9994   1.0000
## Prevalence             0.2851   0.1939   0.1745   0.1631   0.1833
## Detection Rate         0.2845   0.1927   0.1732   0.1626   0.1833
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9988   0.9964   0.9954   0.9977   0.9997
```

#####V.2 Cross Validation Out of Sample Error#####

The out of sample error equals 1 minus accuracy (0.9963)

```r
(1-0.9963)
```

```
## [1] 0.0037
```

#####V.3 Model Evaluation with Test Data#####

The model is evaluated using test data. 

```r
predt <- predict(modFit, testing)
```

The results figure below.

```r
print(predt)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

####VI. Results and Analysis####

The cross validation results show the followings:

1. The model accuracy is 0.9995 with 95% confidence. It is good (>0.8)
2. The out of sample error equals 0.0037 (less than 5%)  
2. The P Value is small (< 2.2e-16), indicating a statistically significant test
3. The Kappa statistic is high (0.9994), suggesting a near complete agreement (close to 1)
4. The sensitivity and specificity are high (>.99%) for each class (A,B,C,D,E).

Details of the analysis for each class sensitivity and specificity are shown above.

####VII. Conclusions####

The Human Activity Recognition (HAR) predictive model is good, as evidenced by the cross validation, and generalized in the 20 test cases. The manner in which the participants will do their exercise can be predicted with 99.95% accuracy and 95% confidence. The HAR random forest predictive model performs well.

####VIII. Appendix####

####VIII.1 Exploratory Data Analysis####


```r
#Training set
str(training)
```

```
## 'data.frame':	13737 obs. of  53 variables:
##  $ roll_belt           : num  1.41 1.42 1.48 1.45 1.42 1.42 1.43 1.45 1.43 1.42 ...
##  $ pitch_belt          : num  8.07 8.07 8.07 8.06 8.09 8.13 8.16 8.17 8.18 8.2 ...
##  $ yaw_belt            : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt    : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ gyros_belt_x        : num  0 0 0.02 0.02 0.02 0.02 0.02 0.03 0.02 0.02 ...
##  $ gyros_belt_y        : num  0 0 0.02 0 0 0 0 0 0 0 ...
##  $ gyros_belt_z        : num  -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 0 -0.02 0 ...
##  $ accel_belt_x        : int  -21 -20 -21 -21 -22 -22 -20 -21 -22 -22 ...
##  $ accel_belt_y        : int  4 5 2 4 3 4 2 4 2 4 ...
##  $ accel_belt_z        : int  22 23 24 21 21 21 24 22 23 21 ...
##  $ magnet_belt_x       : int  -3 -2 -6 0 -4 -2 1 -3 -2 -3 ...
##  $ magnet_belt_y       : int  599 600 600 603 599 603 602 609 602 606 ...
##  $ magnet_belt_z       : int  -313 -305 -302 -312 -311 -313 -312 -308 -319 -309 ...
##  $ roll_arm            : num  -128 -128 -128 -128 -128 -128 -128 -128 -128 -128 ...
##  $ pitch_arm           : num  22.5 22.5 22.1 22 21.9 21.8 21.7 21.6 21.5 21.4 ...
##  $ yaw_arm             : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
##  $ total_accel_arm     : int  34 34 34 34 34 34 34 34 34 34 ...
##  $ gyros_arm_x         : num  0 0.02 0 0.02 0 0.02 0.02 0.02 0.02 0.02 ...
##  $ gyros_arm_y         : num  0 -0.02 -0.03 -0.03 -0.03 -0.02 -0.03 -0.03 -0.03 -0.02 ...
##  $ gyros_arm_z         : num  -0.02 -0.02 0 0 0 0 -0.02 -0.02 0 -0.02 ...
##  $ accel_arm_x         : int  -288 -289 -289 -289 -289 -289 -288 -288 -288 -287 ...
##  $ accel_arm_y         : int  109 110 111 111 111 111 109 110 111 111 ...
##  $ accel_arm_z         : int  -123 -126 -123 -122 -125 -124 -122 -124 -123 -124 ...
##  $ magnet_arm_x        : int  -368 -368 -374 -369 -373 -372 -369 -376 -363 -372 ...
##  $ magnet_arm_y        : int  337 344 337 342 336 338 341 334 343 338 ...
##  $ magnet_arm_z        : int  516 513 506 513 509 510 518 516 520 509 ...
##  $ roll_dumbbell       : num  13.1 12.9 13.4 13.4 13.1 ...
##  $ pitch_dumbbell      : num  -70.5 -70.3 -70.4 -70.8 -70.2 ...
##  $ yaw_dumbbell        : num  -84.9 -85.1 -84.9 -84.5 -85.1 ...
##  $ total_accel_dumbbell: int  37 37 37 37 37 37 37 37 37 37 ...
##  $ gyros_dumbbell_x    : num  0 0 0 0 0 0 0 0 0 0 ...
##  $ gyros_dumbbell_y    : num  -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 ...
##  $ gyros_dumbbell_z    : num  0 0 0 0 0 0 0 0 0 -0.02 ...
##  $ accel_dumbbell_x    : int  -234 -232 -233 -234 -232 -234 -232 -235 -233 -234 ...
##  $ accel_dumbbell_y    : int  47 46 48 48 47 46 47 48 47 48 ...
##  $ accel_dumbbell_z    : int  -271 -270 -270 -269 -270 -272 -269 -270 -270 -269 ...
##  $ magnet_dumbbell_x   : int  -559 -561 -554 -558 -551 -555 -549 -558 -554 -552 ...
##  $ magnet_dumbbell_y   : int  293 298 292 294 295 300 292 291 291 302 ...
##  $ magnet_dumbbell_z   : num  -65 -63 -68 -66 -70 -74 -65 -69 -65 -69 ...
##  $ roll_forearm        : num  28.4 28.3 28 27.9 27.9 27.8 27.7 27.7 27.5 27.2 ...
##  $ pitch_forearm       : num  -63.9 -63.9 -63.9 -63.9 -63.9 -63.8 -63.8 -63.8 -63.8 -63.9 ...
##  $ yaw_forearm         : num  -153 -152 -152 -152 -152 -152 -152 -152 -152 -151 ...
##  $ total_accel_forearm : int  36 36 36 36 36 36 36 36 36 36 ...
##  $ gyros_forearm_x     : num  0.03 0.03 0.02 0.02 0.02 0.02 0.03 0.02 0.02 0 ...
##  $ gyros_forearm_y     : num  0 -0.02 0 -0.02 0 -0.02 0 0 0.02 0 ...
##  $ gyros_forearm_z     : num  -0.02 0 -0.02 -0.03 -0.02 0 -0.02 -0.02 -0.03 -0.03 ...
##  $ accel_forearm_x     : int  192 196 189 193 195 193 193 190 191 193 ...
##  $ accel_forearm_y     : int  203 204 206 203 205 205 204 205 203 205 ...
##  $ accel_forearm_z     : int  -215 -213 -214 -215 -215 -213 -214 -215 -215 -215 ...
##  $ magnet_forearm_x    : int  -17 -18 -17 -9 -18 -9 -16 -22 -11 -15 ...
##  $ magnet_forearm_y    : num  654 658 655 660 659 660 653 656 657 655 ...
##  $ magnet_forearm_z    : num  476 469 473 478 470 474 476 473 478 472 ...
##  $ classe              : Factor w/ 5 levels "A","B","C","D",..: 1 1 1 1 1 1 1 1 1 1 ...
```

```r
head(training)
```

```
##   roll_belt pitch_belt yaw_belt total_accel_belt gyros_belt_x gyros_belt_y
## 1      1.41       8.07    -94.4                3         0.00         0.00
## 3      1.42       8.07    -94.4                3         0.00         0.00
## 5      1.48       8.07    -94.4                3         0.02         0.02
## 6      1.45       8.06    -94.4                3         0.02         0.00
## 7      1.42       8.09    -94.4                3         0.02         0.00
## 8      1.42       8.13    -94.4                3         0.02         0.00
##   gyros_belt_z accel_belt_x accel_belt_y accel_belt_z magnet_belt_x
## 1        -0.02          -21            4           22            -3
## 3        -0.02          -20            5           23            -2
## 5        -0.02          -21            2           24            -6
## 6        -0.02          -21            4           21             0
## 7        -0.02          -22            3           21            -4
## 8        -0.02          -22            4           21            -2
##   magnet_belt_y magnet_belt_z roll_arm pitch_arm yaw_arm total_accel_arm
## 1           599          -313     -128      22.5    -161              34
## 3           600          -305     -128      22.5    -161              34
## 5           600          -302     -128      22.1    -161              34
## 6           603          -312     -128      22.0    -161              34
## 7           599          -311     -128      21.9    -161              34
## 8           603          -313     -128      21.8    -161              34
##   gyros_arm_x gyros_arm_y gyros_arm_z accel_arm_x accel_arm_y accel_arm_z
## 1        0.00        0.00       -0.02        -288         109        -123
## 3        0.02       -0.02       -0.02        -289         110        -126
## 5        0.00       -0.03        0.00        -289         111        -123
## 6        0.02       -0.03        0.00        -289         111        -122
## 7        0.00       -0.03        0.00        -289         111        -125
## 8        0.02       -0.02        0.00        -289         111        -124
##   magnet_arm_x magnet_arm_y magnet_arm_z roll_dumbbell pitch_dumbbell
## 1         -368          337          516      13.05217      -70.49400
## 3         -368          344          513      12.85075      -70.27812
## 5         -374          337          506      13.37872      -70.42856
## 6         -369          342          513      13.38246      -70.81759
## 7         -373          336          509      13.12695      -70.24757
## 8         -372          338          510      12.75083      -70.34768
##   yaw_dumbbell total_accel_dumbbell gyros_dumbbell_x gyros_dumbbell_y
## 1    -84.87394                   37                0            -0.02
## 3    -85.14078                   37                0            -0.02
## 5    -84.85306                   37                0            -0.02
## 6    -84.46500                   37                0            -0.02
## 7    -85.09961                   37                0            -0.02
## 8    -85.09708                   37                0            -0.02
##   gyros_dumbbell_z accel_dumbbell_x accel_dumbbell_y accel_dumbbell_z
## 1                0             -234               47             -271
## 3                0             -232               46             -270
## 5                0             -233               48             -270
## 6                0             -234               48             -269
## 7                0             -232               47             -270
## 8                0             -234               46             -272
##   magnet_dumbbell_x magnet_dumbbell_y magnet_dumbbell_z roll_forearm
## 1              -559               293               -65         28.4
## 3              -561               298               -63         28.3
## 5              -554               292               -68         28.0
## 6              -558               294               -66         27.9
## 7              -551               295               -70         27.9
## 8              -555               300               -74         27.8
##   pitch_forearm yaw_forearm total_accel_forearm gyros_forearm_x
## 1         -63.9        -153                  36            0.03
## 3         -63.9        -152                  36            0.03
## 5         -63.9        -152                  36            0.02
## 6         -63.9        -152                  36            0.02
## 7         -63.9        -152                  36            0.02
## 8         -63.8        -152                  36            0.02
##   gyros_forearm_y gyros_forearm_z accel_forearm_x accel_forearm_y
## 1            0.00           -0.02             192             203
## 3           -0.02            0.00             196             204
## 5            0.00           -0.02             189             206
## 6           -0.02           -0.03             193             203
## 7            0.00           -0.02             195             205
## 8           -0.02            0.00             193             205
##   accel_forearm_z magnet_forearm_x magnet_forearm_y magnet_forearm_z
## 1            -215              -17              654              476
## 3            -213              -18              658              469
## 5            -214              -17              655              473
## 6            -215               -9              660              478
## 7            -215              -18              659              470
## 8            -213               -9              660              474
##   classe
## 1      A
## 3      A
## 5      A
## 6      A
## 7      A
## 8      A
```

```r
summary(training)
```

```
##    roll_belt        pitch_belt          yaw_belt      total_accel_belt
##  Min.   :-28.90   Min.   :-54.9000   Min.   :-180.0   Min.   : 0.0    
##  1st Qu.:  1.10   1st Qu.:  1.7200   1st Qu.: -88.3   1st Qu.: 3.0    
##  Median :113.00   Median :  5.2700   Median : -13.2   Median :17.0    
##  Mean   : 64.34   Mean   :  0.2839   Mean   : -11.2   Mean   :11.3    
##  3rd Qu.:123.00   3rd Qu.: 14.9000   3rd Qu.:  12.9   3rd Qu.:18.0    
##  Max.   :162.00   Max.   : 60.3000   Max.   : 179.0   Max.   :29.0    
##   gyros_belt_x        gyros_belt_y       gyros_belt_z    
##  Min.   :-1.040000   Min.   :-0.53000   Min.   :-1.3500  
##  1st Qu.:-0.030000   1st Qu.: 0.00000   1st Qu.:-0.2000  
##  Median : 0.030000   Median : 0.02000   Median :-0.1000  
##  Mean   :-0.004687   Mean   : 0.03911   Mean   :-0.1321  
##  3rd Qu.: 0.110000   3rd Qu.: 0.11000   3rd Qu.:-0.0200  
##  Max.   : 2.220000   Max.   : 0.63000   Max.   : 1.6100  
##   accel_belt_x       accel_belt_y    accel_belt_z    magnet_belt_x   
##  Min.   :-120.000   Min.   :-69.0   Min.   :-275.0   Min.   :-52.00  
##  1st Qu.: -21.000   1st Qu.:  3.0   1st Qu.:-162.0   1st Qu.:  9.00  
##  Median : -15.000   Median : 33.0   Median :-152.0   Median : 35.00  
##  Mean   :  -5.608   Mean   : 30.1   Mean   : -72.5   Mean   : 55.81  
##  3rd Qu.:  -5.000   3rd Qu.: 61.0   3rd Qu.:  27.0   3rd Qu.: 59.00  
##  Max.   :  85.000   Max.   :164.0   Max.   : 105.0   Max.   :481.00  
##  magnet_belt_y   magnet_belt_z       roll_arm         pitch_arm      
##  Min.   :354.0   Min.   :-623.0   Min.   :-180.00   Min.   :-88.800  
##  1st Qu.:581.0   1st Qu.:-375.0   1st Qu.: -31.90   1st Qu.:-26.000  
##  Median :601.0   Median :-320.0   Median :   0.00   Median :  0.000  
##  Mean   :593.5   Mean   :-345.5   Mean   :  18.06   Mean   : -4.729  
##  3rd Qu.:610.0   3rd Qu.:-306.0   3rd Qu.:  77.60   3rd Qu.: 11.200  
##  Max.   :673.0   Max.   : 293.0   Max.   : 179.00   Max.   : 88.500  
##     yaw_arm           total_accel_arm  gyros_arm_x        gyros_arm_y     
##  Min.   :-180.00000   Min.   : 1.00   Min.   :-6.37000   Min.   :-3.4400  
##  1st Qu.: -42.70000   1st Qu.:17.00   1st Qu.:-1.33000   1st Qu.:-0.8000  
##  Median :   0.00000   Median :27.00   Median : 0.08000   Median :-0.2400  
##  Mean   :  -0.02531   Mean   :25.52   Mean   : 0.05161   Mean   :-0.2594  
##  3rd Qu.:  47.00000   3rd Qu.:33.00   3rd Qu.: 1.59000   3rd Qu.: 0.1400  
##  Max.   : 180.00000   Max.   :65.00   Max.   : 4.87000   Max.   : 2.8400  
##   gyros_arm_z       accel_arm_x       accel_arm_y       accel_arm_z     
##  Min.   :-2.3300   Min.   :-404.00   Min.   :-318.00   Min.   :-630.00  
##  1st Qu.:-0.0700   1st Qu.:-241.00   1st Qu.: -54.00   1st Qu.:-142.00  
##  Median : 0.2500   Median : -44.00   Median :  15.00   Median : -48.00  
##  Mean   : 0.2716   Mean   : -60.55   Mean   :  32.65   Mean   : -71.74  
##  3rd Qu.: 0.7200   3rd Qu.:  84.00   3rd Qu.: 139.00   3rd Qu.:  23.00  
##  Max.   : 2.9500   Max.   : 437.00   Max.   : 303.00   Max.   : 271.00  
##   magnet_arm_x   magnet_arm_y     magnet_arm_z    roll_dumbbell    
##  Min.   :-584   Min.   :-386.0   Min.   :-597.0   Min.   :-153.71  
##  1st Qu.:-300   1st Qu.:  -9.0   1st Qu.: 133.0   1st Qu.: -17.97  
##  Median : 288   Median : 202.0   Median : 444.0   Median :  47.95  
##  Mean   : 191   Mean   : 156.8   Mean   : 305.6   Mean   :  23.98  
##  3rd Qu.: 637   3rd Qu.: 323.0   3rd Qu.: 544.0   3rd Qu.:  67.46  
##  Max.   : 782   Max.   : 583.0   Max.   : 690.0   Max.   : 153.55  
##  pitch_dumbbell     yaw_dumbbell      total_accel_dumbbell
##  Min.   :-148.50   Min.   :-148.766   Min.   : 0.00       
##  1st Qu.: -40.60   1st Qu.: -77.658   1st Qu.: 4.00       
##  Median : -20.67   Median :  -3.191   Median :10.00       
##  Mean   : -10.56   Mean   :   1.895   Mean   :13.68       
##  3rd Qu.:  17.80   3rd Qu.:  80.894   3rd Qu.:19.00       
##  Max.   : 149.40   Max.   : 154.952   Max.   :58.00       
##  gyros_dumbbell_x    gyros_dumbbell_y   gyros_dumbbell_z  
##  Min.   :-204.0000   Min.   :-2.10000   Min.   : -2.3800  
##  1st Qu.:  -0.0300   1st Qu.:-0.14000   1st Qu.: -0.3100  
##  Median :   0.1300   Median : 0.05000   Median : -0.1300  
##  Mean   :   0.1546   Mean   : 0.04623   Mean   : -0.1201  
##  3rd Qu.:   0.3500   3rd Qu.: 0.21000   3rd Qu.:  0.0300  
##  Max.   :   2.2200   Max.   :52.00000   Max.   :317.0000  
##  accel_dumbbell_x  accel_dumbbell_y  accel_dumbbell_z  magnet_dumbbell_x
##  Min.   :-419.00   Min.   :-182.00   Min.   :-284.00   Min.   :-643.0   
##  1st Qu.: -50.00   1st Qu.:  -8.00   1st Qu.:-141.00   1st Qu.:-535.0   
##  Median :  -8.00   Median :  41.00   Median :  -1.00   Median :-479.0   
##  Mean   : -28.27   Mean   :  52.38   Mean   : -37.75   Mean   :-328.6   
##  3rd Qu.:  11.00   3rd Qu.: 110.00   3rd Qu.:  39.00   3rd Qu.:-303.0   
##  Max.   : 235.00   Max.   : 315.00   Max.   : 318.00   Max.   : 583.0   
##  magnet_dumbbell_y magnet_dumbbell_z  roll_forearm     pitch_forearm   
##  Min.   :-3600.0   Min.   :-262.00   Min.   :-180.00   Min.   :-72.50  
##  1st Qu.:  230.0   1st Qu.: -46.00   1st Qu.:  -1.93   1st Qu.:  0.00  
##  Median :  310.0   Median :  13.00   Median :  20.80   Median :  9.31  
##  Mean   :  221.2   Mean   :  46.59   Mean   :  33.24   Mean   : 10.75  
##  3rd Qu.:  390.0   3rd Qu.:  96.00   3rd Qu.: 140.00   3rd Qu.: 28.60  
##  Max.   :  632.0   Max.   : 451.00   Max.   : 180.00   Max.   : 89.80  
##   yaw_forearm      total_accel_forearm gyros_forearm_x   
##  Min.   :-180.00   Min.   :  0.00      Min.   :-22.0000  
##  1st Qu.: -69.10   1st Qu.: 29.00      1st Qu.: -0.2200  
##  Median :   0.00   Median : 36.00      Median :  0.0500  
##  Mean   :  18.87   Mean   : 34.75      Mean   :  0.1523  
##  3rd Qu.: 110.00   3rd Qu.: 41.00      3rd Qu.:  0.5600  
##  Max.   : 180.00   Max.   :108.00      Max.   :  3.9700  
##  gyros_forearm_y     gyros_forearm_z    accel_forearm_x   accel_forearm_y 
##  Min.   : -7.02000   Min.   : -8.0900   Min.   :-498.00   Min.   :-595.0  
##  1st Qu.: -1.46000   1st Qu.: -0.1800   1st Qu.:-179.00   1st Qu.:  53.0  
##  Median :  0.03000   Median :  0.0800   Median : -57.00   Median : 201.0  
##  Mean   :  0.09408   Mean   :  0.1587   Mean   : -62.36   Mean   : 163.5  
##  3rd Qu.:  1.65000   3rd Qu.:  0.4900   3rd Qu.:  75.00   3rd Qu.: 313.0  
##  Max.   :311.00000   Max.   :231.0000   Max.   : 477.00   Max.   : 923.0  
##  accel_forearm_z   magnet_forearm_x  magnet_forearm_y magnet_forearm_z
##  Min.   :-446.00   Min.   :-1280.0   Min.   :-896.0   Min.   :-973.0  
##  1st Qu.:-181.00   1st Qu.: -617.0   1st Qu.:   1.0   1st Qu.: 183.0  
##  Median : -37.00   Median : -380.0   Median : 592.0   Median : 508.0  
##  Mean   : -54.13   Mean   : -314.2   Mean   : 380.1   Mean   : 391.4  
##  3rd Qu.:  26.00   3rd Qu.:  -75.0   3rd Qu.: 738.0   3rd Qu.: 653.0  
##  Max.   : 291.00   Max.   :  663.0   Max.   :1480.0   Max.   :1090.0  
##  classe  
##  A:3906  
##  B:2658  
##  C:2396  
##  D:2252  
##  E:2525  
## 
```

```r
#Cross Validation set
str(validation)
```

```
## 'data.frame':	5885 obs. of  53 variables:
##  $ roll_belt           : num  1.41 1.48 1.45 1.6 1.54 1.52 1.43 1.34 1.33 1.31 ...
##  $ pitch_belt          : num  8.07 8.05 8.18 8.1 8.11 8.16 8.17 8.05 7.76 7.69 ...
##  $ yaw_belt            : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.3 -94.2 -94.2 ...
##  $ total_accel_belt    : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ gyros_belt_x        : num  0.02 0.02 0.03 0.02 0.03 0.03 0 0 -0.02 0.02 ...
##  $ gyros_belt_y        : num  0 0 0 0 0 0 0 0.02 0.02 0 ...
##  $ gyros_belt_z        : num  -0.02 -0.03 -0.02 -0.02 -0.02 -0.02 -0.03 -0.03 0 -0.03 ...
##  $ accel_belt_x        : int  -22 -22 -21 -20 -22 -20 -22 -22 -19 -19 ...
##  $ accel_belt_y        : int  4 3 2 1 3 4 4 3 4 3 ...
##  $ accel_belt_z        : int  22 21 23 20 22 23 19 20 21 21 ...
##  $ magnet_belt_x       : int  -7 -6 -5 -10 3 -4 4 2 -3 -2 ...
##  $ magnet_belt_y       : int  608 604 596 607 597 606 602 602 601 597 ...
##  $ magnet_belt_z       : int  -311 -310 -317 -304 -320 -320 -316 -317 -318 -317 ...
##  $ roll_arm            : num  -128 -128 -128 -129 -129 -129 -129 -130 -130 -130 ...
##  $ pitch_arm           : num  22.5 22.1 21.5 20.9 20.7 20.7 20.5 19.9 19.7 19.6 ...
##  $ yaw_arm             : num  -161 -161 -161 -161 -161 -161 -161 -162 -162 -162 ...
##  $ total_accel_arm     : int  34 34 34 34 34 34 34 34 34 34 ...
##  $ gyros_arm_x         : num  0.02 0.02 0.02 0.03 -0.02 -0.02 0.03 0.02 0 -0.02 ...
##  $ gyros_arm_y         : num  -0.02 -0.03 -0.03 -0.02 -0.02 -0.02 -0.02 -0.03 -0.03 -0.02 ...
##  $ gyros_arm_z         : num  -0.02 0.02 0 -0.02 0 0 0 -0.02 -0.02 -0.03 ...
##  $ accel_arm_x         : int  -290 -289 -290 -288 -289 -290 -290 -289 -289 -288 ...
##  $ accel_arm_y         : int  110 111 110 111 111 109 110 110 111 108 ...
##  $ accel_arm_z         : int  -125 -123 -123 -124 -126 -125 -126 -124 -124 -124 ...
##  $ magnet_arm_x        : int  -369 -372 -366 -375 -371 -367 -375 -372 -370 -371 ...
##  $ magnet_arm_y        : int  337 344 339 337 338 337 339 339 332 337 ...
##  $ magnet_arm_z        : int  513 512 509 513 505 514 508 507 506 508 ...
##  $ roll_dumbbell       : num  13.1 13.4 13.1 13.4 12.8 ...
##  $ pitch_dumbbell      : num  -70.6 -70.4 -70.6 -70.8 -70.5 ...
##  $ yaw_dumbbell        : num  -84.7 -84.9 -84.7 -84.5 -84.9 ...
##  $ total_accel_dumbbell: int  37 37 37 37 37 37 37 37 37 37 ...
##  $ gyros_dumbbell_x    : num  0 0 0 0 0 0 0 -0.02 0 0 ...
##  $ gyros_dumbbell_y    : num  -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 0 -0.02 0 ...
##  $ gyros_dumbbell_z    : num  0 -0.02 0 0 0 -0.02 -0.02 0 0 0 ...
##  $ accel_dumbbell_x    : int  -233 -232 -233 -234 -233 -234 -234 -232 -233 -235 ...
##  $ accel_dumbbell_y    : int  47 48 47 48 46 47 48 49 47 47 ...
##  $ accel_dumbbell_z    : int  -269 -269 -269 -269 -270 -271 -272 -269 -269 -267 ...
##  $ magnet_dumbbell_x   : int  -555 -552 -564 -554 -556 -552 -556 -554 -558 -554 ...
##  $ magnet_dumbbell_y   : int  296 303 299 299 297 291 298 297 290 295 ...
##  $ magnet_dumbbell_z   : num  -64 -60 -64 -72 -68 -60 -62 -62 -65 -61 ...
##  $ roll_forearm        : num  28.3 28.1 27.6 26.9 27 26.8 26.7 26.4 26 25.5 ...
##  $ pitch_forearm       : num  -63.9 -63.9 -63.8 -63.9 -63.6 -63.6 -63.7 -63.9 -63.9 -63.8 ...
##  $ yaw_forearm         : num  -153 -152 -152 -151 -151 -151 -151 -150 -150 -149 ...
##  $ total_accel_forearm : int  36 36 36 36 36 36 36 36 36 36 ...
##  $ gyros_forearm_x     : num  0.02 0.02 0.02 0.03 0.03 0.02 0 0.02 0.03 0.02 ...
##  $ gyros_forearm_y     : num  0 -0.02 -0.02 -0.03 0 -0.02 -0.02 -0.02 0 0 ...
##  $ gyros_forearm_z     : num  -0.02 0 -0.02 -0.02 0 -0.03 -0.02 -0.02 0 -0.07 ...
##  $ accel_forearm_x     : int  192 189 193 194 192 195 196 192 191 195 ...
##  $ accel_forearm_y     : int  203 206 205 208 204 205 207 207 202 207 ...
##  $ accel_forearm_z     : int  -216 -214 -214 -214 -214 -217 -216 -216 -214 -216 ...
##  $ magnet_forearm_x    : int  -18 -16 -17 -11 -14 -12 -15 -13 -10 -15 ...
##  $ magnet_forearm_y    : num  661 658 657 654 657 657 650 659 655 657 ...
##  $ magnet_forearm_z    : num  473 469 465 469 467 469 473 465 469 460 ...
##  $ classe              : Factor w/ 5 levels "A","B","C","D",..: 1 1 1 1 1 1 1 1 1 1 ...
```

```r
head(validation)
```

```
##    roll_belt pitch_belt yaw_belt total_accel_belt gyros_belt_x
## 2       1.41       8.07    -94.4                3         0.02
## 4       1.48       8.05    -94.4                3         0.02
## 11      1.45       8.18    -94.4                3         0.03
## 21      1.60       8.10    -94.4                3         0.02
## 27      1.54       8.11    -94.4                3         0.03
## 29      1.52       8.16    -94.4                3         0.03
##    gyros_belt_y gyros_belt_z accel_belt_x accel_belt_y accel_belt_z
## 2             0        -0.02          -22            4           22
## 4             0        -0.03          -22            3           21
## 11            0        -0.02          -21            2           23
## 21            0        -0.02          -20            1           20
## 27            0        -0.02          -22            3           22
## 29            0        -0.02          -20            4           23
##    magnet_belt_x magnet_belt_y magnet_belt_z roll_arm pitch_arm yaw_arm
## 2             -7           608          -311     -128      22.5    -161
## 4             -6           604          -310     -128      22.1    -161
## 11            -5           596          -317     -128      21.5    -161
## 21           -10           607          -304     -129      20.9    -161
## 27             3           597          -320     -129      20.7    -161
## 29            -4           606          -320     -129      20.7    -161
##    total_accel_arm gyros_arm_x gyros_arm_y gyros_arm_z accel_arm_x
## 2               34        0.02       -0.02       -0.02        -290
## 4               34        0.02       -0.03        0.02        -289
## 11              34        0.02       -0.03        0.00        -290
## 21              34        0.03       -0.02       -0.02        -288
## 27              34       -0.02       -0.02        0.00        -289
## 29              34       -0.02       -0.02        0.00        -290
##    accel_arm_y accel_arm_z magnet_arm_x magnet_arm_y magnet_arm_z
## 2          110        -125         -369          337          513
## 4          111        -123         -372          344          512
## 11         110        -123         -366          339          509
## 21         111        -124         -375          337          513
## 27         111        -126         -371          338          505
## 29         109        -125         -367          337          514
##    roll_dumbbell pitch_dumbbell yaw_dumbbell total_accel_dumbbell
## 2       13.13074      -70.63751    -84.71065                   37
## 4       13.43120      -70.39379    -84.87363                   37
## 11      13.13074      -70.63751    -84.71065                   37
## 21      13.38246      -70.81759    -84.46500                   37
## 27      12.82749      -70.49032    -84.93557                   37
## 29      13.05217      -70.49400    -84.87394                   37
##    gyros_dumbbell_x gyros_dumbbell_y gyros_dumbbell_z accel_dumbbell_x
## 2                 0            -0.02             0.00             -233
## 4                 0            -0.02            -0.02             -232
## 11                0            -0.02             0.00             -233
## 21                0            -0.02             0.00             -234
## 27                0            -0.02             0.00             -233
## 29                0            -0.02            -0.02             -234
##    accel_dumbbell_y accel_dumbbell_z magnet_dumbbell_x magnet_dumbbell_y
## 2                47             -269              -555               296
## 4                48             -269              -552               303
## 11               47             -269              -564               299
## 21               48             -269              -554               299
## 27               46             -270              -556               297
## 29               47             -271              -552               291
##    magnet_dumbbell_z roll_forearm pitch_forearm yaw_forearm
## 2                -64         28.3         -63.9        -153
## 4                -60         28.1         -63.9        -152
## 11               -64         27.6         -63.8        -152
## 21               -72         26.9         -63.9        -151
## 27               -68         27.0         -63.6        -151
## 29               -60         26.8         -63.6        -151
##    total_accel_forearm gyros_forearm_x gyros_forearm_y gyros_forearm_z
## 2                   36            0.02            0.00           -0.02
## 4                   36            0.02           -0.02            0.00
## 11                  36            0.02           -0.02           -0.02
## 21                  36            0.03           -0.03           -0.02
## 27                  36            0.03            0.00            0.00
## 29                  36            0.02           -0.02           -0.03
##    accel_forearm_x accel_forearm_y accel_forearm_z magnet_forearm_x
## 2              192             203            -216              -18
## 4              189             206            -214              -16
## 11             193             205            -214              -17
## 21             194             208            -214              -11
## 27             192             204            -214              -14
## 29             195             205            -217              -12
##    magnet_forearm_y magnet_forearm_z classe
## 2               661              473      A
## 4               658              469      A
## 11              657              465      A
## 21              654              469      A
## 27              657              467      A
## 29              657              469      A
```

```r
summary(validation)
```

```
##    roll_belt        pitch_belt          yaw_belt       total_accel_belt
##  Min.   :-28.30   Min.   :-55.8000   Min.   :-178.00   Min.   : 1.00   
##  1st Qu.:  1.09   1st Qu.:  1.8800   1st Qu.: -88.30   1st Qu.: 3.00   
##  Median :114.00   Median :  5.3100   Median : -12.30   Median :17.00   
##  Mean   : 64.57   Mean   :  0.3551   Mean   : -11.22   Mean   :11.34   
##  3rd Qu.:123.00   3rd Qu.: 15.2000   3rd Qu.:  12.80   3rd Qu.:18.00   
##  Max.   :162.00   Max.   : 60.1000   Max.   : 179.00   Max.   :28.00   
##   gyros_belt_x        gyros_belt_y       gyros_belt_z    
##  Min.   :-1.000000   Min.   :-0.64000   Min.   :-1.4600  
##  1st Qu.:-0.030000   1st Qu.: 0.00000   1st Qu.:-0.2000  
##  Median : 0.030000   Median : 0.02000   Median :-0.1000  
##  Mean   :-0.007704   Mean   : 0.04071   Mean   :-0.1268  
##  3rd Qu.: 0.110000   3rd Qu.: 0.11000   3rd Qu.:-0.0200  
##  Max.   : 2.020000   Max.   : 0.64000   Max.   : 1.6200  
##   accel_belt_x      accel_belt_y     accel_belt_z     magnet_belt_x   
##  Min.   :-75.000   Min.   :-54.00   Min.   :-268.00   Min.   :-46.00  
##  1st Qu.:-21.000   1st Qu.:  3.00   1st Qu.:-162.00   1st Qu.:  9.00  
##  Median :-15.000   Median : 36.00   Median :-153.00   Median : 35.00  
##  Mean   : -5.564   Mean   : 30.29   Mean   : -72.81   Mean   : 55.12  
##  3rd Qu.: -5.000   3rd Qu.: 61.00   3rd Qu.:  28.00   3rd Qu.: 59.00  
##  Max.   : 81.000   Max.   :109.00   Max.   : 104.00   Max.   :485.00  
##  magnet_belt_y   magnet_belt_z       roll_arm        pitch_arm      
##  Min.   :365.0   Min.   :-616.0   Min.   :-180.0   Min.   :-88.200  
##  1st Qu.:582.0   1st Qu.:-375.0   1st Qu.: -31.0   1st Qu.:-25.200  
##  Median :601.0   Median :-319.0   Median :   0.0   Median :  0.000  
##  Mean   :594.1   Mean   :-345.5   Mean   :  17.3   Mean   : -4.338  
##  3rd Qu.:610.0   3rd Qu.:-305.0   3rd Qu.:  76.5   3rd Qu.: 11.200  
##  Max.   :668.0   Max.   : 284.0   Max.   : 180.0   Max.   : 86.400  
##     yaw_arm         total_accel_arm  gyros_arm_x        gyros_arm_y     
##  Min.   :-180.000   Min.   : 1.00   Min.   :-6.36000   Min.   :-3.4000  
##  1st Qu.: -43.600   1st Qu.:17.00   1st Qu.:-1.32000   1st Qu.:-0.7900  
##  Median :   0.000   Median :27.00   Median : 0.06000   Median :-0.2400  
##  Mean   :  -2.004   Mean   :25.49   Mean   : 0.02215   Mean   :-0.2517  
##  3rd Qu.:  42.600   3rd Qu.:32.00   3rd Qu.: 1.54000   3rd Qu.: 0.1400  
##  Max.   : 180.000   Max.   :66.00   Max.   : 4.87000   Max.   : 2.8100  
##   gyros_arm_z       accel_arm_x       accel_arm_y       accel_arm_z     
##  Min.   :-2.1000   Min.   :-364.00   Min.   :-302.00   Min.   :-636.00  
##  1st Qu.:-0.0800   1st Qu.:-242.00   1st Qu.: -53.00   1st Qu.:-144.00  
##  Median : 0.2300   Median : -44.00   Median :  11.00   Median : -44.00  
##  Mean   : 0.2646   Mean   : -59.51   Mean   :  32.48   Mean   : -70.09  
##  3rd Qu.: 0.7200   3rd Qu.:  83.00   3rd Qu.: 140.00   3rd Qu.:  25.00  
##  Max.   : 3.0200   Max.   : 431.00   Max.   : 308.00   Max.   : 292.00  
##   magnet_arm_x     magnet_arm_y     magnet_arm_z    roll_dumbbell    
##  Min.   :-578.0   Min.   :-392.0   Min.   :-590.0   Min.   :-153.51  
##  1st Qu.:-299.0   1st Qu.:  -8.0   1st Qu.: 129.0   1st Qu.: -19.31  
##  Median : 290.0   Median : 199.0   Median : 444.0   Median :  48.80  
##  Mean   : 193.5   Mean   : 156.2   Mean   : 308.5   Mean   :  23.51  
##  3rd Qu.: 638.0   3rd Qu.: 323.0   3rd Qu.: 546.0   3rd Qu.:  68.12  
##  Max.   : 778.0   Max.   : 577.0   Max.   : 694.0   Max.   : 153.38  
##  pitch_dumbbell     yaw_dumbbell      total_accel_dumbbell
##  Min.   :-149.59   Min.   :-150.871   Min.   : 0.00       
##  1st Qu.: -41.50   1st Qu.: -77.529   1st Qu.: 4.00       
##  Median : -21.70   Median :  -3.375   Median :11.00       
##  Mean   : -11.28   Mean   :   1.159   Mean   :13.81       
##  3rd Qu.:  16.85   3rd Qu.:  77.418   3rd Qu.:20.00       
##  Max.   : 129.82   Max.   : 154.754   Max.   :42.00       
##  gyros_dumbbell_x  gyros_dumbbell_y   gyros_dumbbell_z  accel_dumbbell_x 
##  Min.   :-1.9900   Min.   :-1.96000   Min.   :-2.3000   Min.   :-237.00  
##  1st Qu.:-0.0200   1st Qu.:-0.14000   1st Qu.:-0.3300   1st Qu.: -51.00  
##  Median : 0.1400   Median : 0.03000   Median :-0.1300   Median :  -9.00  
##  Mean   : 0.1763   Mean   : 0.04566   Mean   :-0.1498   Mean   : -29.43  
##  3rd Qu.: 0.3700   3rd Qu.: 0.21000   3rd Qu.: 0.0300   3rd Qu.:  10.00  
##  Max.   : 2.2000   Max.   : 4.37000   Max.   : 1.6700   Max.   : 224.00  
##  accel_dumbbell_y  accel_dumbbell_z  magnet_dumbbell_x magnet_dumbbell_y
##  Min.   :-189.00   Min.   :-334.00   Min.   :-639.0    Min.   :-744.0   
##  1st Qu.:  -9.00   1st Qu.:-143.00   1st Qu.:-535.0    1st Qu.: 233.0   
##  Median :  44.00   Median :  -1.00   Median :-479.0    Median : 311.0   
##  Mean   :  53.22   Mean   : -39.66   Mean   :-328.3    Mean   : 220.4   
##  3rd Qu.: 114.00   3rd Qu.:  37.00   3rd Qu.:-307.0    3rd Qu.: 391.0   
##  Max.   : 300.00   Max.   : 318.00   Max.   : 592.0    Max.   : 633.0   
##  magnet_dumbbell_z  roll_forearm    pitch_forearm     yaw_forearm     
##  Min.   :-249.00   Min.   :-180.0   Min.   :-72.50   Min.   :-180.00  
##  1st Qu.: -44.00   1st Qu.:   0.0   1st Qu.:  0.00   1st Qu.: -67.40  
##  Median :  14.00   Median :  22.8   Median :  9.04   Median :   0.00  
##  Mean   :  44.79   Mean   :  35.2   Mean   : 10.61   Mean   :  19.99  
##  3rd Qu.:  95.00   3rd Qu.: 141.0   3rd Qu.: 27.60   3rd Qu.: 110.00  
##  Max.   : 452.00   Max.   : 180.0   Max.   : 88.70   Max.   : 180.00  
##  total_accel_forearm gyros_forearm_x   gyros_forearm_y   
##  Min.   : 0.00       Min.   :-2.9700   Min.   :-6.62000  
##  1st Qu.:29.00       1st Qu.:-0.2100   1st Qu.:-1.49000  
##  Median :35.00       Median : 0.0500   Median : 0.02000  
##  Mean   :34.64       Mean   : 0.1711   Mean   : 0.03104  
##  3rd Qu.:41.00       3rd Qu.: 0.5800   3rd Qu.: 1.57000  
##  Max.   :78.00       Max.   : 3.1000   Max.   : 6.13000  
##  gyros_forearm_z   accel_forearm_x   accel_forearm_y accel_forearm_z  
##  Min.   :-3.2300   Min.   :-477.00   Min.   :-632    Min.   :-386.00  
##  1st Qu.:-0.1600   1st Qu.:-176.00   1st Qu.:  63    1st Qu.:-182.00  
##  Median : 0.0800   Median : -57.00   Median : 200    Median : -45.00  
##  Mean   : 0.1338   Mean   : -59.99   Mean   : 164    Mean   : -58.01  
##  3rd Qu.: 0.4800   3rd Qu.:  78.00   3rd Qu.: 311    3rd Qu.:  25.00  
##  Max.   : 4.3100   Max.   : 375.00   Max.   : 591    Max.   : 287.00  
##  magnet_forearm_x  magnet_forearm_y magnet_forearm_z classe  
##  Min.   :-1280.0   Min.   :-890.0   Min.   :-966.0   A:1674  
##  1st Qu.: -611.0   1st Qu.:   6.0   1st Qu.: 204.0   B:1139  
##  Median : -374.0   Median : 590.0   Median : 517.0   C:1026  
##  Mean   : -308.7   Mean   : 380.1   Mean   : 398.7   D: 964  
##  3rd Qu.:  -72.0   3rd Qu.: 734.0   3rd Qu.: 653.0   E:1082  
##  Max.   :  672.0   Max.   :1450.0   Max.   :1070.0
```

```r
#Testing set
str(testing)
```

```
## 'data.frame':	20 obs. of  53 variables:
##  $ roll_belt           : num  123 1.02 0.87 125 1.35 -5.92 1.2 0.43 0.93 114 ...
##  $ pitch_belt          : num  27 4.87 1.82 -41.6 3.33 1.59 4.44 4.15 6.72 22.4 ...
##  $ yaw_belt            : num  -4.75 -88.9 -88.5 162 -88.6 -87.7 -87.3 -88.5 -93.7 -13.1 ...
##  $ total_accel_belt    : int  20 4 5 17 3 4 4 4 4 18 ...
##  $ gyros_belt_x        : num  -0.5 -0.06 0.05 0.11 0.03 0.1 -0.06 -0.18 0.1 0.14 ...
##  $ gyros_belt_y        : num  -0.02 -0.02 0.02 0.11 0.02 0.05 0 -0.02 0 0.11 ...
##  $ gyros_belt_z        : num  -0.46 -0.07 0.03 -0.16 0 -0.13 0 -0.03 -0.02 -0.16 ...
##  $ accel_belt_x        : int  -38 -13 1 46 -8 -11 -14 -10 -15 -25 ...
##  $ accel_belt_y        : int  69 11 -1 45 4 -16 2 -2 1 63 ...
##  $ accel_belt_z        : int  -179 39 49 -156 27 38 35 42 32 -158 ...
##  $ magnet_belt_x       : int  -13 43 29 169 33 31 50 39 -6 10 ...
##  $ magnet_belt_y       : int  581 636 631 608 566 638 622 635 600 601 ...
##  $ magnet_belt_z       : int  -382 -309 -312 -304 -418 -291 -315 -305 -302 -330 ...
##  $ roll_arm            : num  40.7 0 0 -109 76.1 0 0 0 -137 -82.4 ...
##  $ pitch_arm           : num  -27.8 0 0 55 2.76 0 0 0 11.2 -63.8 ...
##  $ yaw_arm             : num  178 0 0 -142 102 0 0 0 -167 -75.3 ...
##  $ total_accel_arm     : int  10 38 44 25 29 14 15 22 34 32 ...
##  $ gyros_arm_x         : num  -1.65 -1.17 2.1 0.22 -1.96 0.02 2.36 -3.71 0.03 0.26 ...
##  $ gyros_arm_y         : num  0.48 0.85 -1.36 -0.51 0.79 0.05 -1.01 1.85 -0.02 -0.5 ...
##  $ gyros_arm_z         : num  -0.18 -0.43 1.13 0.92 -0.54 -0.07 0.89 -0.69 -0.02 0.79 ...
##  $ accel_arm_x         : int  16 -290 -341 -238 -197 -26 99 -98 -287 -301 ...
##  $ accel_arm_y         : int  38 215 245 -57 200 130 79 175 111 -42 ...
##  $ accel_arm_z         : int  93 -90 -87 6 -30 -19 -67 -78 -122 -80 ...
##  $ magnet_arm_x        : int  -326 -325 -264 -173 -170 396 702 535 -367 -420 ...
##  $ magnet_arm_y        : int  385 447 474 257 275 176 15 215 335 294 ...
##  $ magnet_arm_z        : int  481 434 413 633 617 516 217 385 520 493 ...
##  $ roll_dumbbell       : num  -17.7 54.5 57.1 43.1 -101.4 ...
##  $ pitch_dumbbell      : num  25 -53.7 -51.4 -30 -53.4 ...
##  $ yaw_dumbbell        : num  126.2 -75.5 -75.2 -103.3 -14.2 ...
##  $ total_accel_dumbbell: int  9 31 29 18 4 29 29 29 3 2 ...
##  $ gyros_dumbbell_x    : num  0.64 0.34 0.39 0.1 0.29 -0.59 0.34 0.37 0.03 0.42 ...
##  $ gyros_dumbbell_y    : num  0.06 0.05 0.14 -0.02 -0.47 0.8 0.16 0.14 -0.21 0.51 ...
##  $ gyros_dumbbell_z    : num  -0.61 -0.71 -0.34 0.05 -0.46 1.1 -0.23 -0.39 -0.21 -0.03 ...
##  $ accel_dumbbell_x    : int  21 -153 -141 -51 -18 -138 -145 -140 0 -7 ...
##  $ accel_dumbbell_y    : int  -15 155 155 72 -30 166 150 159 25 -20 ...
##  $ accel_dumbbell_z    : int  81 -205 -196 -148 -5 -186 -190 -191 9 7 ...
##  $ magnet_dumbbell_x   : int  523 -502 -506 -576 -424 -543 -484 -515 -519 -531 ...
##  $ magnet_dumbbell_y   : int  -528 388 349 238 252 262 354 350 348 321 ...
##  $ magnet_dumbbell_z   : int  -56 -36 41 53 312 96 97 53 -32 -164 ...
##  $ roll_forearm        : num  141 109 131 0 -176 150 155 -161 15.5 13.2 ...
##  $ pitch_forearm       : num  49.3 -17.6 -32.6 0 -2.16 1.46 34.5 43.6 -63.5 19.4 ...
##  $ yaw_forearm         : num  156 106 93 0 -47.9 89.7 152 -89.5 -139 -105 ...
##  $ total_accel_forearm : int  33 39 34 43 24 43 32 47 36 24 ...
##  $ gyros_forearm_x     : num  0.74 1.12 0.18 1.38 -0.75 -0.88 -0.53 0.63 0.03 0.02 ...
##  $ gyros_forearm_y     : num  -3.34 -2.78 -0.79 0.69 3.1 4.26 1.8 -0.74 0.02 0.13 ...
##  $ gyros_forearm_z     : num  -0.59 -0.18 0.28 1.8 0.8 1.35 0.75 0.49 -0.02 -0.07 ...
##  $ accel_forearm_x     : int  -110 212 154 -92 131 230 -192 -151 195 -212 ...
##  $ accel_forearm_y     : int  267 297 271 406 -93 322 170 -331 204 98 ...
##  $ accel_forearm_z     : int  -149 -118 -129 -39 172 -144 -175 -282 -217 -7 ...
##  $ magnet_forearm_x    : int  -714 -237 -51 -233 375 -300 -678 -109 0 -403 ...
##  $ magnet_forearm_y    : int  419 791 698 783 -787 800 284 -619 652 723 ...
##  $ magnet_forearm_z    : int  617 873 783 521 91 884 585 -32 469 512 ...
##  $ problem_id          : int  1 2 3 4 5 6 7 8 9 10 ...
```

```r
head(testing)
```

```
##   roll_belt pitch_belt yaw_belt total_accel_belt gyros_belt_x gyros_belt_y
## 1    123.00      27.00    -4.75               20        -0.50        -0.02
## 2      1.02       4.87   -88.90                4        -0.06        -0.02
## 3      0.87       1.82   -88.50                5         0.05         0.02
## 4    125.00     -41.60   162.00               17         0.11         0.11
## 5      1.35       3.33   -88.60                3         0.03         0.02
## 6     -5.92       1.59   -87.70                4         0.10         0.05
##   gyros_belt_z accel_belt_x accel_belt_y accel_belt_z magnet_belt_x
## 1        -0.46          -38           69         -179           -13
## 2        -0.07          -13           11           39            43
## 3         0.03            1           -1           49            29
## 4        -0.16           46           45         -156           169
## 5         0.00           -8            4           27            33
## 6        -0.13          -11          -16           38            31
##   magnet_belt_y magnet_belt_z roll_arm pitch_arm yaw_arm total_accel_arm
## 1           581          -382     40.7    -27.80     178              10
## 2           636          -309      0.0      0.00       0              38
## 3           631          -312      0.0      0.00       0              44
## 4           608          -304   -109.0     55.00    -142              25
## 5           566          -418     76.1      2.76     102              29
## 6           638          -291      0.0      0.00       0              14
##   gyros_arm_x gyros_arm_y gyros_arm_z accel_arm_x accel_arm_y accel_arm_z
## 1       -1.65        0.48       -0.18          16          38          93
## 2       -1.17        0.85       -0.43        -290         215         -90
## 3        2.10       -1.36        1.13        -341         245         -87
## 4        0.22       -0.51        0.92        -238         -57           6
## 5       -1.96        0.79       -0.54        -197         200         -30
## 6        0.02        0.05       -0.07         -26         130         -19
##   magnet_arm_x magnet_arm_y magnet_arm_z roll_dumbbell pitch_dumbbell
## 1         -326          385          481     -17.73748       24.96085
## 2         -325          447          434      54.47761      -53.69758
## 3         -264          474          413      57.07031      -51.37303
## 4         -173          257          633      43.10927      -30.04885
## 5         -170          275          617    -101.38396      -53.43952
## 6          396          176          516      62.18750      -50.55595
##   yaw_dumbbell total_accel_dumbbell gyros_dumbbell_x gyros_dumbbell_y
## 1    126.23596                    9             0.64             0.06
## 2    -75.51480                   31             0.34             0.05
## 3    -75.20287                   29             0.39             0.14
## 4   -103.32003                   18             0.10            -0.02
## 5    -14.19542                    4             0.29            -0.47
## 6    -71.12063                   29            -0.59             0.80
##   gyros_dumbbell_z accel_dumbbell_x accel_dumbbell_y accel_dumbbell_z
## 1            -0.61               21              -15               81
## 2            -0.71             -153              155             -205
## 3            -0.34             -141              155             -196
## 4             0.05              -51               72             -148
## 5            -0.46              -18              -30               -5
## 6             1.10             -138              166             -186
##   magnet_dumbbell_x magnet_dumbbell_y magnet_dumbbell_z roll_forearm
## 1               523              -528               -56          141
## 2              -502               388               -36          109
## 3              -506               349                41          131
## 4              -576               238                53            0
## 5              -424               252               312         -176
## 6              -543               262                96          150
##   pitch_forearm yaw_forearm total_accel_forearm gyros_forearm_x
## 1         49.30       156.0                  33            0.74
## 2        -17.60       106.0                  39            1.12
## 3        -32.60        93.0                  34            0.18
## 4          0.00         0.0                  43            1.38
## 5         -2.16       -47.9                  24           -0.75
## 6          1.46        89.7                  43           -0.88
##   gyros_forearm_y gyros_forearm_z accel_forearm_x accel_forearm_y
## 1           -3.34           -0.59            -110             267
## 2           -2.78           -0.18             212             297
## 3           -0.79            0.28             154             271
## 4            0.69            1.80             -92             406
## 5            3.10            0.80             131             -93
## 6            4.26            1.35             230             322
##   accel_forearm_z magnet_forearm_x magnet_forearm_y magnet_forearm_z
## 1            -149             -714              419              617
## 2            -118             -237              791              873
## 3            -129              -51              698              783
## 4             -39             -233              783              521
## 5             172              375             -787               91
## 6            -144             -300              800              884
##   problem_id
## 1          1
## 2          2
## 3          3
## 4          4
## 5          5
## 6          6
```

```r
summary(testing)
```

```
##    roll_belt          pitch_belt         yaw_belt      total_accel_belt
##  Min.   : -5.9200   Min.   :-41.600   Min.   :-93.70   Min.   : 2.00   
##  1st Qu.:  0.9075   1st Qu.:  3.013   1st Qu.:-88.62   1st Qu.: 3.00   
##  Median :  1.1100   Median :  4.655   Median :-87.85   Median : 4.00   
##  Mean   : 31.3055   Mean   :  5.824   Mean   :-59.30   Mean   : 7.55   
##  3rd Qu.: 32.5050   3rd Qu.:  6.135   3rd Qu.:-63.50   3rd Qu.: 8.00   
##  Max.   :129.0000   Max.   : 27.800   Max.   :162.00   Max.   :21.00   
##   gyros_belt_x     gyros_belt_y     gyros_belt_z      accel_belt_x   
##  Min.   :-0.500   Min.   :-0.050   Min.   :-0.4800   Min.   :-48.00  
##  1st Qu.:-0.070   1st Qu.:-0.005   1st Qu.:-0.1375   1st Qu.:-19.00  
##  Median : 0.020   Median : 0.000   Median :-0.0250   Median :-13.00  
##  Mean   :-0.045   Mean   : 0.010   Mean   :-0.1005   Mean   :-13.50  
##  3rd Qu.: 0.070   3rd Qu.: 0.020   3rd Qu.: 0.0000   3rd Qu.: -8.75  
##  Max.   : 0.240   Max.   : 0.110   Max.   : 0.0500   Max.   : 46.00  
##   accel_belt_y     accel_belt_z     magnet_belt_x    magnet_belt_y  
##  Min.   :-16.00   Min.   :-187.00   Min.   :-13.00   Min.   :566.0  
##  1st Qu.:  2.00   1st Qu.: -24.00   1st Qu.:  5.50   1st Qu.:578.5  
##  Median :  4.50   Median :  27.00   Median : 33.50   Median :600.5  
##  Mean   : 18.35   Mean   : -17.60   Mean   : 35.15   Mean   :601.5  
##  3rd Qu.: 25.50   3rd Qu.:  38.25   3rd Qu.: 46.25   3rd Qu.:631.2  
##  Max.   : 72.00   Max.   :  49.00   Max.   :169.00   Max.   :638.0  
##  magnet_belt_z       roll_arm         pitch_arm          yaw_arm       
##  Min.   :-426.0   Min.   :-137.00   Min.   :-63.800   Min.   :-167.00  
##  1st Qu.:-398.5   1st Qu.:   0.00   1st Qu.: -9.188   1st Qu.: -60.15  
##  Median :-313.5   Median :   0.00   Median :  0.000   Median :   0.00  
##  Mean   :-346.9   Mean   :  16.42   Mean   : -3.950   Mean   :  -2.80  
##  3rd Qu.:-305.0   3rd Qu.:  71.53   3rd Qu.:  3.465   3rd Qu.:  25.50  
##  Max.   :-291.0   Max.   : 152.00   Max.   : 55.000   Max.   : 178.00  
##  total_accel_arm  gyros_arm_x      gyros_arm_y       gyros_arm_z     
##  Min.   : 3.00   Min.   :-3.710   Min.   :-2.0900   Min.   :-0.6900  
##  1st Qu.:20.25   1st Qu.:-0.645   1st Qu.:-0.6350   1st Qu.:-0.1800  
##  Median :29.50   Median : 0.020   Median :-0.0400   Median :-0.0250  
##  Mean   :26.40   Mean   : 0.077   Mean   :-0.1595   Mean   : 0.1205  
##  3rd Qu.:33.25   3rd Qu.: 1.248   3rd Qu.: 0.2175   3rd Qu.: 0.5650  
##  Max.   :44.00   Max.   : 3.660   Max.   : 1.8500   Max.   : 1.1300  
##   accel_arm_x      accel_arm_y      accel_arm_z       magnet_arm_x    
##  Min.   :-341.0   Min.   :-65.00   Min.   :-404.00   Min.   :-428.00  
##  1st Qu.:-277.0   1st Qu.: 52.25   1st Qu.:-128.50   1st Qu.:-373.75  
##  Median :-194.5   Median :112.00   Median : -83.50   Median :-265.00  
##  Mean   :-134.6   Mean   :103.10   Mean   : -87.85   Mean   : -38.95  
##  3rd Qu.:   5.5   3rd Qu.:168.25   3rd Qu.: -27.25   3rd Qu.: 250.50  
##  Max.   : 106.0   Max.   :245.00   Max.   :  93.00   Max.   : 750.00  
##   magnet_arm_y     magnet_arm_z    roll_dumbbell      pitch_dumbbell  
##  Min.   :-307.0   Min.   :-499.0   Min.   :-111.118   Min.   :-54.97  
##  1st Qu.: 205.2   1st Qu.: 403.0   1st Qu.:   7.494   1st Qu.:-51.89  
##  Median : 291.0   Median : 476.5   Median :  50.403   Median :-40.81  
##  Mean   : 239.4   Mean   : 369.8   Mean   :  33.760   Mean   :-19.47  
##  3rd Qu.: 358.8   3rd Qu.: 517.0   3rd Qu.:  58.129   3rd Qu.: 16.12  
##  Max.   : 474.0   Max.   : 633.0   Max.   : 123.984   Max.   : 96.87  
##   yaw_dumbbell       total_accel_dumbbell gyros_dumbbell_x 
##  Min.   :-103.3200   Min.   : 1.0         Min.   :-1.0300  
##  1st Qu.: -75.2809   1st Qu.: 7.0         1st Qu.: 0.1600  
##  Median :  -8.2863   Median :15.5         Median : 0.3600  
##  Mean   :  -0.9385   Mean   :17.2         Mean   : 0.2690  
##  3rd Qu.:  55.8335   3rd Qu.:29.0         3rd Qu.: 0.4625  
##  Max.   : 132.2337   Max.   :31.0         Max.   : 1.0600  
##  gyros_dumbbell_y  gyros_dumbbell_z accel_dumbbell_x  accel_dumbbell_y
##  Min.   :-1.1100   Min.   :-1.180   Min.   :-159.00   Min.   :-30.00  
##  1st Qu.:-0.2100   1st Qu.:-0.485   1st Qu.:-140.25   1st Qu.:  5.75  
##  Median : 0.0150   Median :-0.280   Median : -19.00   Median : 71.50  
##  Mean   : 0.0605   Mean   :-0.266   Mean   : -47.60   Mean   : 70.55  
##  3rd Qu.: 0.1450   3rd Qu.:-0.165   3rd Qu.:  15.75   3rd Qu.:151.25  
##  Max.   : 1.9100   Max.   : 1.100   Max.   : 185.00   Max.   :166.00  
##  accel_dumbbell_z magnet_dumbbell_x magnet_dumbbell_y magnet_dumbbell_z
##  Min.   :-221.0   Min.   :-576.0    Min.   :-558.0    Min.   :-164.00  
##  1st Qu.:-192.2   1st Qu.:-528.0    1st Qu.: 259.5    1st Qu.: -33.00  
##  Median :  -3.0   Median :-508.5    Median : 316.0    Median :  49.50  
##  Mean   : -60.0   Mean   :-304.2    Mean   : 189.3    Mean   :  71.40  
##  3rd Qu.:  76.5   3rd Qu.:-317.0    3rd Qu.: 348.2    3rd Qu.:  96.25  
##  Max.   : 100.0   Max.   : 523.0    Max.   : 403.0    Max.   : 368.00  
##   roll_forearm     pitch_forearm      yaw_forearm      
##  Min.   :-176.00   Min.   :-63.500   Min.   :-168.000  
##  1st Qu.: -40.25   1st Qu.:-11.457   1st Qu.: -93.375  
##  Median :  94.20   Median :  8.830   Median : -19.250  
##  Mean   :  38.66   Mean   :  7.099   Mean   :   2.195  
##  3rd Qu.: 143.25   3rd Qu.: 28.500   3rd Qu.: 104.500  
##  Max.   : 176.00   Max.   : 59.300   Max.   : 159.000  
##  total_accel_forearm gyros_forearm_x   gyros_forearm_y   gyros_forearm_z  
##  Min.   :21.00       Min.   :-1.0600   Min.   :-5.9700   Min.   :-1.2600  
##  1st Qu.:24.00       1st Qu.:-0.5850   1st Qu.:-1.2875   1st Qu.:-0.0975  
##  Median :32.50       Median : 0.0200   Median : 0.0350   Median : 0.2300  
##  Mean   :32.05       Mean   :-0.0200   Mean   :-0.0415   Mean   : 0.2610  
##  3rd Qu.:36.75       3rd Qu.: 0.2925   3rd Qu.: 2.0475   3rd Qu.: 0.7625  
##  Max.   :47.00       Max.   : 1.3800   Max.   : 4.2600   Max.   : 1.8000  
##  accel_forearm_x  accel_forearm_y  accel_forearm_z  magnet_forearm_x
##  Min.   :-212.0   Min.   :-331.0   Min.   :-282.0   Min.   :-714.0  
##  1st Qu.:-114.8   1st Qu.:   8.5   1st Qu.:-199.0   1st Qu.:-427.2  
##  Median :  86.0   Median : 138.0   Median :-148.5   Median :-189.5  
##  Mean   :  38.8   Mean   : 125.3   Mean   : -93.7   Mean   :-159.2  
##  3rd Qu.: 166.2   3rd Qu.: 268.0   3rd Qu.: -31.0   3rd Qu.:  41.5  
##  Max.   : 232.0   Max.   : 406.0   Max.   : 179.0   Max.   : 532.0  
##  magnet_forearm_y magnet_forearm_z   problem_id   
##  Min.   :-787.0   Min.   :-32.0    Min.   : 1.00  
##  1st Qu.:-328.8   1st Qu.:275.2    1st Qu.: 5.75  
##  Median : 487.0   Median :491.5    Median :10.50  
##  Mean   : 191.8   Mean   :460.2    Mean   :10.50  
##  3rd Qu.: 720.8   3rd Qu.:661.5    3rd Qu.:15.25  
##  Max.   : 800.0   Max.   :884.0    Max.   :20.00
```

####VIII.2 Cross Validation Results####

```r
print(predcv)
```

```
##     2     4    11    21    27    29    33    43    47    49    52    63 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##    64    68    70    71    72    74    79    84    85    90    95   104 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   105   107   108   109   110   111   112   117   118   121   122   135 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   139   141   142   143   145   146   152   156   158   165   166   167 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   175   182   195   197   199   200   202   208   212   213   218   219 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   221   223   227   228   231   232   233   235   236   240   242   246 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   247   248   249   253   255   256   263   268   269   270   274   275 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   279   282   284   287   291   292   294   297   299   300   302   310 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   311   315   318   320   330   336   345   348   356   357   359   361 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   365   368   371   383   384   387   389   390   391   397   398   401 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   409   414   416   417   420   423   427   430   442   444   445   446 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   450   455   457   462   464   466   476   477   478   480   484   485 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   487   489   492   493   504   506   508   511   512   516   520   521 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   524   530   537   538   540   543   546   548   550   551   552   553 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   556   558   559   561   574   581   582   584   585   587   592   594 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   595   602   603   604   605   607   614   615   620   624   631   633 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   635   636   644   645   649   651   655   656   657   661   674   680 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   685   689   690   695   705   714   715   716   719   721   723   724 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   727   729   734   739   740   742   743   745   753   756   762   763 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   764   765   767   768   773   774   776   789   795   797   801   803 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   805   810   813   821   822   826   829   833   834   837   839   840 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   843   846   848   850   851   864   865   872   874   878   879   884 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   887   888   890   891   893   896   902   904   905   907   917   919 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   920   922   923   925   927   928   932   937   942   944   945   946 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   948   955   956   960   972   982   989   990  1000  1001  1002  1006 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1008  1010  1012  1013  1015  1019  1020  1021  1022  1025  1026  1028 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1031  1037  1042  1043  1046  1048  1051  1054  1058  1068  1072  1078 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1079  1080  1089  1091  1092  1095  1096  1102  1105  1111  1118  1128 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1132  1133  1136  1137  1141  1142  1143  1153  1154  1155  1157  1161 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1164  1165  1167  1168  1172  1176  1179  1183  1186  1187  1192  1201 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1206  1208  1215  1218  1221  1223  1224  1226  1233  1238  1249  1258 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1262  1269  1271  1280  1286  1289  1292  1293  1296  1300  1301  1305 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1311  1313  1314  1315  1318  1319  1320  1325  1326  1327  1330  1331 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1333  1335  1344  1346  1348  1352  1361  1362  1365  1367  1369  1371 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1372  1373  1375  1379  1387  1392  1399  1400  1407  1409  1410  1412 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1420  1428  1434  1439  1441  1454  1457  1458  1459  1462  1466  1468 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1475  1478  1479  1481  1484  1488  1489  1500  1504  1505  1506  1508 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1509  1510  1511  1514  1518  1524  1527  1529  1540  1541  1542  1545 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1546  1547  1548  1549  1553  1554  1557  1566  1568  1574  1576  1583 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1584  1588  1590  1591  1595  1597  1599  1601  1608  1612  1614  1616 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1619  1621  1624  1628  1632  1637  1640  1645  1651  1667  1670  1672 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1673  1674  1678  1681  1686  1694  1698  1703  1706  1708  1714  1719 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1720  1723  1724  1727  1729  1732  1733  1735  1737  1748  1749  1750 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1751  1752  1753  1754  1756  1758  1761  1764  1766  1770  1772  1788 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1790  1791  1799  1806  1807  1808  1812  1814  1817  1821  1824  1836 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1838  1839  1846  1852  1853  1854  1857  1859  1863  1865  1867  1871 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1872  1874  1886  1891  1892  1893  1894  1896  1898  1899  1902  1903 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1911  1914  1916  1925  1926  1929  1930  1931  1932  1935  1940  1945 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1948  1950  1952  1953  1958  1959  1962  1964  1969  1972  1976  1986 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1993  1994  1996  1998  2000  2004  2005  2011  2013  2014  2015  2018 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2020  2024  2025  2028  2031  2033  2034  2035  2036  2041  2044  2052 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2054  2055  2069  2075  2077  2079  2081  2086  2087  2088  2091  2095 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2097  2098  2099  2104  2106  2107  2113  2115  2117  2118  2119  2121 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2122  2123  2125  2126  2127  2128  2130  2132  2133  2134  2140  2143 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2150  2155  2157  2169  2171  2175  2187  2188  2189  2191  2197  2198 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2199  2209  2211  2212  2214  2219  2220  2229  2230  2235  2237  2248 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2252  2258  2259  2261  2273  2276  2278  2279  2283  2284  2287  2289 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2293  2303  2305  2311  2319  2328  2330  2333  2335  2337  2339  2341 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2342  2344  2345  2348  2349  2352  2354  2360  2361  2362  2367  2371 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2375  2377  2378  2389  2390  2393  2397  2398  2400  2403  2407  2409 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2410  2417  2424  2429  2431  2433  2434  2435  2437  2444  2451  2454 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2457  2458  2460  2469  2477  2483  2486  2487  2489  2493  2495  2498 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2500  2503  2506  2507  2513  2514  2515  2519  2520  2524  2526  2527 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2528  2533  2534  2538  2546  2547  2556  2560  2563  2566  2571  2573 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2575  2576  2579  2584  2585  2594  2595  2601  2609  2610  2612  2613 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2615  2617  2618  2626  2633  2636  2639  2641  2643  2646  2648  2652 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2657  2662  2663  2664  2666  2674  2675  2676  2682  2687  2691  2692 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2694  2695  2697  2702  2703  2704  2706  2708  2711  2714  2716  2720 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2721  2728  2732  2736  2740  2742  2745  2746  2749  2757  2764  2766 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2768  2770  2771  2772  2778  2780  2781  2795  2796  2806  2807  2808 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2810  2813  2816  2817  2818  2819  2826  2829  2830  2831  2836  2845 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2846  2847  2853  2854  2862  2864  2870  2871  2876  2877  2882  2883 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2885  2888  2893  2894  2896  2903  2907  2910  2912  2914  2916  2918 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2921  2924  2925  2927  2930  2931  2938  2945  2946  2950  2958  2960 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2964  2969  2975  2979  2989  2991  2993  2994  2998  3000  3004  3007 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3013  3015  3019  3023  3024  3026  3030  3031  3035  3041  3042  3044 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3048  3049  3052  3053  3060  3068  3069  3071  3072  3075  3079  3080 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3084  3086  3087  3093  3100  3101  3102  3105  3111  3115  3118  3119 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3123  3124  3129  3130  3135  3143  3144  3145  3147  3152  3157  3166 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3168  3172  3174  3177  3181  3185  3188  3192  3196  3197  3199  3200 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3202  3203  3205  3207  3212  3213  3214  3216  3223  3232  3233  3242 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3246  3248  3249  3255  3260  3263  3269  3271  3273  3278  3280  3281 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3283  3285  3288  3290  3291  3293  3296  3298  3304  3308  3309  3310 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3311  3313  3315  3316  3320  3325  3332  3336  3340  3342  3343  3344 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3346  3347  3348  3349  3354  3356  3357  3359  3363  3364  3365  3367 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3370  3371  3372  3373  3376  3377  3384  3392  3395  3396  3398  3400 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3401  3405  3406  3407  3408  3421  3422  3424  3425  3428  3429  3430 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3431  3432  3433  3434  3436  3438  3440  3448  3450  3452  3453  3454 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3457  3458  3459  3462  3466  3482  3487  3493  3502  3505  3511  3513 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3514  3517  3521  3527  3529  3530  3531  3533  3536  3540  3541  3546 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3553  3554  3555  3558  3561  3562  3567  3569  3572  3574  3581  3583 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3584  3585  3587  3591  3597  3600  3601  3608  3612  3616  3620  3622 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3625  3633  3638  3639  3640  3642  3643  3647  3651  3653  3661  3665 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3669  3677  3681  3682  3683  3685  3690  3691  3696  3698  3702  3706 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3708  3709  3711  3712  3713  3717  3721  3723  3727  3728  3729  3733 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3740  3742  3743  3749  3752  3754  3761  3766  3768  3771  3772  3773 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3780  3788  3796  3800  3804  3816  3827  3830  3831  3832  3835  3840 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3841  3843  3846  3847  3853  3856  3858  3859  3862  3863  3864  3867 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3879  3882  3885  3890  3891  3893  3894  3905  3906  3914  3915  3920 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3926  3927  3931  3940  3943  3950  3952  3954  3955  3961  3962  3967 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3975  3978  3980  3983  3984  3985  3988  3994  4006  4010  4013  4014 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4017  4020  4028  4030  4032  4038  4040  4041  4043  4049  4050  4057 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4062  4066  4067  4069  4074  4076  4079  4085  4087  4088  4091  4097 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4101  4102  4104  4105  4107  4108  4111  4114  4116  4118  4121  4128 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4133  4134  4138  4144  4150  4151  4153  4154  4158  4159  4160  4163 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4166  4170  4176  4182  4183  4184  4188  4189  4192  4204  4205  4214 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4215  4216  4219  4221  4226  4228  4239  4240  4241  4244  4246  4247 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4249  4250  4251  4258  4263  4268  4270  4278  4279  4282  4286  4288 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4289  4292  4295  4296  4298  4299  4315  4318  4326  4329  4330  4331 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4333  4334  4348  4356  4358  4359  4361  4364  4366  4369  4371  4375 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4376  4377  4383  4384  4386  4389  4391  4392  4396  4401  4402  4403 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4404  4407  4413  4427  4428  4435  4441  4446  4447  4449  4451  4455 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4457  4458  4465  4471  4473  4475  4476  4482  4483  4486  4488  4489 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4490  4495  4496  4500  4501  4506  4513  4514  4515  4522  4523  4524 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4525  4529  4532  4535  4539  4541  4543  4545  4549  4551  4555  4556 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4558  4561  4564  4566  4567  4569  4571  4573  4580  4583  4585  4586 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4587  4591  4597  4602  4603  4608  4610  4612  4613  4615  4616  4618 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4621  4624  4630  4632  4634  4644  4646  4648  4650  4653  4655  4659 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4661  4665  4666  4674  4679  4691  4693  4696  4701  4708  4709  4710 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4713  4715  4722  4725  4728  4735  4742  4753  4754  4758  4764  4765 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4768  4771  4774  4775  4778  4785  4789  4790  4793  4794  4797  4801 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4802  4805  4806  4808  4813  4816  4820  4821  4823  4825  4827  4829 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4834  4837  4844  4847  4853  4857  4863  4865  4866  4868  4874  4879 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4882  4883  4888  4889  4890  4895  4898  4899  4902  4903  4905  4908 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4911  4914  4916  4917  4918  4920  4925  4936  4939  4943  4944  4946 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4957  4961  4966  4968  4969  4971  4978  4979  4981  4984  4985  4987 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4990  4994  4996  5002  5003  5005  5008  5009  5011  5016  5019  5021 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5025  5027  5030  5036  5037  5049  5052  5060  5062  5063  5064  5066 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5072  5073  5075  5084  5085  5087  5090  5094  5098  5105  5106  5109 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5111  5114  5123  5128  5135  5142  5143  5149  5153  5158  5159  5164 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5166  5170  5171  5174  5177  5182  5185  5186  5190  5192  5198  5199 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5202  5205  5207  5216  5218  5223  5229  5234  5241  5246  5250  5251 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5252  5256  5263  5264  5277  5280  5283  5284  5289  5291  5295  5296 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5299  5301  5303  5306  5307  5311  5312  5313  5317  5323  5325  5328 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5338  5339  5340  5346  5347  5348  5352  5353  5356  5364  5365  5368 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5369  5370  5374  5375  5377  5378  5379  5380  5381  5384  5387  5393 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5395  5410  5412  5417  5423  5425  5430  5432  5435  5437  5438  5442 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5453  5464  5469  5483  5484  5486  5491  5493  5495  5500  5514  5515 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5516  5517  5523  5524  5525  5533  5534  5541  5546  5548  5558  5560 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5561  5562  5571  5574  5577  5578  5590  5591  5599  5600  5604  5605 
##     A     A     A     A     A     A     B     B     B     B     B     B 
##  5606  5613  5616  5618  5624  5626  5627  5636  5637  5642  5644  5648 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  5649  5650  5652  5654  5655  5656  5657  5665  5666  5681  5682  5688 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  5691  5694  5702  5703  5704  5707  5709  5714  5721  5728  5729  5733 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  5734  5740  5742  5744  5749  5750  5751  5752  5753  5759  5761  5764 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  5765  5766  5770  5772  5776  5779  5781  5785  5788  5789  5795  5796 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  5798  5806  5807  5809  5810  5814  5815  5824  5825  5830  5831  5840 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  5843  5844  5848  5849  5851  5853  5858  5862  5864  5866  5874  5879 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  5880  5884  5885  5886  5894  5897  5903  5907  5911  5914  5918  5919 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  5921  5929  5935  5938  5940  5941  5946  5948  5951  5957  5969  5971 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  5972  5974  5976  5981  5984  5986  5987  5988  5989  5990  5992  5998 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6001  6003  6010  6016  6018  6023  6024  6025  6036  6037  6040  6042 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6048  6054  6056  6058  6062  6063  6064  6067  6068  6082  6085  6088 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6098  6101  6106  6107  6111  6115  6120  6121  6122  6124  6128  6129 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6131  6132  6143  6144  6145  6149  6156  6158  6159  6169  6173  6174 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6177  6180  6182  6188  6192  6195  6196  6197  6208  6209  6210  6219 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6221  6222  6224  6227  6229  6230  6231  6234  6239  6242  6245  6248 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6253  6254  6257  6265  6268  6270  6274  6275  6278  6281  6282  6287 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6288  6292  6294  6295  6302  6309  6313  6315  6316  6317  6318  6325 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6326  6327  6331  6333  6335  6343  6345  6347  6351  6354  6355  6374 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6376  6381  6382  6385  6388  6391  6401  6402  6410  6411  6412  6413 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6415  6416  6417  6418  6425  6429  6431  6433  6437  6438  6440  6441 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6449  6453  6454  6455  6457  6460  6461  6467  6468  6469  6471  6475 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6477  6480  6481  6486  6487  6489  6492  6493  6496  6502  6508  6520 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6521  6523  6533  6534  6540  6545  6546  6551  6561  6565  6568  6575 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6576  6582  6583  6584  6587  6588  6590  6591  6594  6602  6603  6604 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6605  6606  6610  6613  6615  6616  6621  6625  6626  6629  6637  6641 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6647  6653  6655  6656  6658  6659  6660  6663  6670  6677  6679  6689 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6690  6692  6693  6708  6718  6720  6724  6725  6729  6730  6734  6735 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6739  6741  6743  6745  6746  6749  6750  6754  6757  6758  6761  6762 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6766  6768  6769  6773  6776  6777  6779  6781  6786  6790  6792  6794 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6796  6805  6807  6813  6816  6823  6825  6828  6830  6831  6833  6839 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6840  6841  6842  6848  6851  6852  6857  6858  6864  6879  6883  6886 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6890  6895  6897  6901  6902  6912  6914  6916  6917  6922  6929  6930 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6938  6946  6947  6948  6950  6952  6959  6965  6968  6970  6976  6980 
##     B     A     A     B     B     B     B     B     B     B     B     B 
##  6981  6984  6988  6995  7000  7002  7018  7022  7031  7038  7044  7049 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7050  7055  7057  7059  7060  7061  7063  7066  7069  7074  7077  7081 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7085  7089  7093  7094  7098  7102  7103  7105  7108  7110  7111  7112 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7119  7120  7124  7126  7135  7138  7139  7141  7142  7146  7148  7154 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7155  7160  7166  7168  7170  7171  7175  7177  7183  7184  7185  7186 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7198  7199  7201  7202  7206  7216  7217  7218  7220  7222  7223  7231 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7236  7238  7239  7241  7249  7250  7251  7252  7259  7260  7263  7268 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7279  7285  7286  7289  7294  7296  7300  7303  7305  7306  7311  7313 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7317  7319  7321  7322  7324  7328  7329  7334  7336  7338  7339  7344 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7347  7354  7355  7356  7358  7360  7361  7362  7363  7375  7379  7380 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7382  7385  7386  7388  7392  7395  7398  7399  7400  7405  7409  7412 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7415  7416  7421  7423  7424  7426  7427  7430  7432  7434  7436  7444 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7445  7446  7447  7450  7451  7464  7466  7473  7474  7477  7487  7488 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7492  7493  7495  7496  7500  7501  7509  7511  7518  7520  7521  7523 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7524  7526  7538  7541  7543  7550  7557  7560  7565  7566  7567  7570 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7579  7597  7602  7603  7613  7617  7621  7627  7629  7631  7632  7633 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7638  7640  7642  7643  7646  7652  7654  7657  7658  7659  7660  7662 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7678  7680  7681  7683  7684  7685  7687  7688  7689  7690  7693  7695 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7702  7707  7713  7714  7718  7719  7728  7731  7732  7741  7744  7749 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7750  7751  7756  7758  7760  7762  7766  7772  7773  7781  7783  7789 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7792  7794  7799  7801  7805  7806  7810  7811  7813  7815  7816  7817 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7819  7825  7826  7832  7836  7840  7843  7845  7848  7849  7852  7860 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7861  7863  7866  7874  7875  7879  7888  7892  7896  7897  7902  7904 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7907  7909  7916  7917  7918  7925  7928  7930  7932  7940  7942  7943 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7944  7945  7948  7953  7957  7958  7977  7980  7983  7985  7988  7989 
##     B     B     B     B     A     A     B     B     B     B     B     B 
##  7997  8001  8002  8004  8012  8016  8021  8024  8025  8027  8038  8040 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8046  8054  8057  8060  8061  8063  8068  8069  8070  8072  8079  8080 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8082  8084  8086  8089  8090  8101  8104  8106  8111  8113  8115  8118 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8123  8125  8131  8132  8134  8135  8138  8142  8143  8144  8145  8149 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8151  8152  8153  8154  8156  8159  8161  8163  8164  8165  8173  8174 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8181  8182  8184  8194  8198  8200  8201  8203  8204  8213  8225  8227 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8230  8233  8235  8236  8238  8242  8246  8251  8259  8263  8268  8274 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8276  8281  8283  8288  8293  8295  8296  8298  8299  8301  8302  8303 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8305  8308  8310  8311  8312  8319  8320  8329  8331  8332  8337  8339 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8342  8343  8344  8346  8348  8349  8350  8352  8360  8361  8362  8368 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8371  8373  8376  8381  8383  8387  8388  8392  8398  8403  8405  8407 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8408  8411  8412  8414  8421  8422  8425  8427  8428  8436  8442  8453 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8454  8457  8460  8471  8475  8484  8485  8486  8491  8492  8494  8496 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8501  8508  8513  8516  8519  8521  8523  8524  8530  8536  8537  8539 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8541  8547  8548  8553  8556  8558  8564  8565  8568  8571  8575  8587 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8591  8598  8600  8605  8607  8608  8614  8615  8616  8622  8626  8629 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8630  8632  8635  8640  8641  8650  8651  8657  8659  8665  8666  8671 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8678  8684  8687  8689  8694  8695  8699  8701  8703  8717  8719  8722 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8727  8735  8736  8740  8741  8754  8756  8760  8763  8766  8771  8774 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8778  8780  8782  8784  8785  8789  8790  8796  8797  8798  8803  8804 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8805  8807  8815  8817  8821  8822  8836  8839  8840  8845  8847  8848 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8851  8860  8865  8870  8871  8872  8873  8874  8876  8877  8879  8884 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8885  8887  8888  8889  8895  8897  8902  8909  8910  8912  8914  8915 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8918  8921  8926  8930  8932  8933  8934  8937  8939  8941  8942  8943 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8947  8950  8953  8954  8958  8959  8960  8963  8967  8969  8974  8978 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8980  8985  8987  8989  9000  9003  9007  9009  9012  9019  9020  9022 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  9024  9034  9036  9037  9046  9056  9060  9062  9063  9069  9072  9073 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  9075  9077  9078  9079  9087  9088  9089  9092  9095  9098  9099  9102 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  9105  9109  9110  9111  9118  9120  9121  9123  9126  9132  9133  9134 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  9135  9149  9151  9152  9153  9162  9163  9164  9165  9167  9169  9171 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  9172  9180  9181  9183  9184  9189  9193  9210  9221  9222  9223  9226 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  9227  9233  9235  9237  9241  9246  9252  9258  9259  9261  9263  9265 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  9270  9271  9276  9280  9282  9286  9294  9297  9298  9300  9302  9304 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  9306  9307  9309  9310  9311  9312  9314  9317  9322  9323  9325  9326 
##     B     B     B     B     B     C     B     B     B     B     B     B 
##  9330  9331  9333  9339  9341  9344  9345  9351  9353  9358  9361  9364 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  9365  9366  9367  9373  9374  9378  9379  9383  9385  9387  9399  9400 
##     B     B     B     B     B     B     C     C     C     C     C     C 
##  9404  9405  9407  9411  9416  9427  9430  9436  9439  9440  9441  9443 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9445  9447  9451  9452  9453  9454  9455  9456  9457  9462  9464  9465 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9467  9469  9475  9478  9481  9482  9484  9485  9487  9488  9490  9493 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9494  9495  9496  9504  9505  9510  9511  9512  9513  9514  9519  9520 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9530  9535  9538  9539  9542  9545  9548  9550  9556  9563  9564  9565 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9568  9569  9570  9573  9575  9578  9584  9586  9589  9596  9597  9604 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9606  9607  9620  9628  9631  9635  9636  9637  9639  9645  9646  9656 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9664  9667  9669  9673  9677  9678  9684  9685  9686  9692  9693  9696 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9702  9708  9709  9710  9712  9713  9716  9717  9718  9726  9730  9731 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9734  9735  9737  9747  9755  9759  9774  9775  9777  9782  9783  9784 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9791  9792  9797  9799  9802  9806  9807  9812  9818  9819  9822  9824 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9829  9830  9837  9838  9841  9842  9846  9848  9850  9858  9864  9870 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9875  9876  9881  9885  9887  9889  9891  9892  9893  9898  9899  9905 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9908  9909  9910  9912  9913  9916  9918  9922  9924  9925  9926  9928 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9940  9942  9944  9946  9947  9950  9951  9952  9954  9958  9965  9967 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9972  9973  9976  9978  9979  9980  9982  9985  9986  9990  9991  9995 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10000 10011 10019 10020 10024 10031 10036 10044 10046 10048 10063 10078 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10079 10101 10103 10104 10111 10114 10117 10121 10123 10125 10126 10128 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10132 10134 10140 10141 10144 10145 10147 10149 10152 10154 10155 10166 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10169 10171 10176 10179 10190 10198 10199 10202 10204 10208 10209 10210 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10211 10213 10215 10227 10228 10229 10231 10232 10234 10235 10243 10254 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10255 10257 10258 10262 10267 10271 10278 10279 10284 10289 10290 10291 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10293 10294 10298 10299 10305 10308 10309 10311 10319 10321 10324 10328 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10331 10338 10345 10349 10353 10359 10362 10364 10371 10373 10375 10376 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10379 10380 10382 10389 10390 10394 10398 10405 10406 10411 10424 10427 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10428 10429 10431 10434 10437 10442 10446 10459 10463 10465 10467 10469 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10471 10472 10476 10477 10478 10480 10482 10483 10484 10487 10489 10495 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10499 10500 10505 10509 10513 10514 10516 10517 10519 10530 10535 10539 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10541 10542 10549 10551 10555 10556 10567 10568 10569 10572 10574 10577 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10578 10579 10581 10588 10589 10590 10591 10595 10597 10599 10601 10603 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10607 10608 10610 10621 10622 10630 10631 10632 10634 10638 10640 10641 
##     C     C     C     C     C     B     B     B     C     C     C     C 
## 10644 10645 10648 10649 10651 10656 10666 10670 10673 10675 10677 10678 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10682 10684 10686 10689 10691 10699 10702 10706 10707 10708 10709 10711 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10712 10714 10717 10723 10725 10726 10728 10734 10736 10740 10742 10744 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10747 10753 10756 10762 10764 10766 10768 10775 10777 10778 10779 10781 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10782 10787 10788 10789 10794 10795 10802 10803 10808 10811 10812 10815 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10824 10826 10833 10835 10837 10839 10843 10844 10851 10852 10858 10860 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10861 10864 10865 10868 10874 10879 10884 10885 10894 10898 10902 10903 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10904 10905 10913 10919 10923 10925 10933 10940 10943 10948 10949 10950 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10953 10956 10957 10958 10964 10965 10967 10969 10976 10978 10979 10981 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10986 10991 10994 10998 11000 11005 11007 11008 11013 11014 11018 11021 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11035 11037 11038 11048 11050 11051 11064 11068 11073 11075 11080 11094 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11099 11100 11101 11102 11103 11110 11113 11117 11119 11124 11127 11128 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11134 11135 11136 11139 11142 11146 11147 11148 11149 11156 11157 11166 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11167 11168 11170 11176 11177 11178 11180 11184 11185 11188 11189 11190 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11192 11195 11197 11202 11205 11206 11208 11212 11213 11216 11221 11223 
##     C     C     C     C     C     C     B     C     C     C     C     C 
## 11224 11230 11232 11233 11236 11241 11242 11252 11254 11255 11263 11273 
##     C     C     C     B     C     C     C     C     C     C     C     C 
## 11274 11275 11277 11278 11284 11287 11290 11298 11301 11307 11314 11316 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11317 11327 11330 11331 11333 11335 11336 11341 11342 11345 11357 11359 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11360 11362 11365 11369 11370 11373 11374 11375 11376 11383 11385 11388 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11389 11393 11395 11402 11404 11409 11410 11413 11415 11425 11430 11432 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11433 11437 11440 11441 11442 11444 11445 11454 11457 11462 11467 11468 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11477 11483 11484 11490 11492 11493 11496 11499 11503 11511 11512 11522 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11524 11525 11528 11534 11538 11539 11541 11547 11550 11556 11558 11559 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11566 11567 11570 11572 11581 11583 11585 11589 11594 11597 11598 11599 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11601 11617 11618 11621 11625 11626 11628 11632 11635 11638 11647 11648 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11653 11658 11659 11665 11667 11668 11672 11673 11675 11676 11678 11687 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11691 11693 11697 11698 11700 11702 11707 11709 11710 11711 11712 11713 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11714 11716 11717 11718 11721 11722 11728 11732 11737 11740 11741 11743 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11751 11752 11754 11757 11762 11765 11766 11769 11770 11771 11773 11775 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11777 11779 11782 11785 11788 11791 11793 11795 11796 11797 11802 11808 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11812 11814 11815 11817 11821 11834 11839 11842 11843 11844 11845 11850 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11852 11855 11857 11860 11861 11870 11871 11872 11875 11878 11881 11883 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11884 11887 11892 11893 11894 11895 11897 11900 11915 11929 11940 11942 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11945 11946 11947 11948 11951 11957 11959 11961 11962 11964 11971 11981 
##     C     C     C     C     C     C     B     C     C     C     C     C 
## 11983 11984 11990 11992 12008 12013 12018 12020 12023 12027 12032 12035 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12037 12041 12045 12046 12047 12051 12056 12057 12061 12064 12065 12079 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12083 12084 12086 12090 12095 12096 12097 12099 12101 12102 12108 12109 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12111 12117 12118 12121 12122 12124 12125 12126 12128 12129 12130 12136 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12144 12149 12151 12152 12153 12155 12157 12164 12166 12169 12171 12172 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12176 12177 12180 12187 12190 12191 12192 12199 12205 12206 12207 12210 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12214 12222 12224 12232 12234 12249 12252 12253 12256 12257 12261 12273 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12278 12286 12289 12290 12292 12293 12298 12299 12300 12303 12311 12312 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12314 12315 12319 12322 12323 12327 12329 12331 12334 12348 12352 12362 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12366 12367 12368 12370 12376 12379 12381 12385 12391 12395 12407 12410 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12412 12415 12416 12417 12425 12426 12428 12430 12431 12434 12437 12443 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12455 12462 12467 12468 12470 12471 12472 12476 12477 12478 12480 12481 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12487 12489 12491 12493 12502 12504 12517 12518 12519 12527 12530 12532 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12537 12540 12541 12544 12546 12553 12558 12560 12562 12575 12578 12580 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12581 12582 12585 12586 12590 12592 12593 12594 12596 12599 12600 12607 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12616 12617 12618 12623 12633 12636 12638 12639 12644 12646 12651 12654 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12655 12656 12657 12662 12664 12665 12666 12667 12668 12674 12675 12677 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12682 12683 12689 12692 12698 12699 12703 12706 12712 12715 12716 12718 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12722 12723 12725 12727 12737 12741 12744 12745 12747 12748 12749 12751 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12752 12758 12759 12769 12770 12773 12776 12783 12791 12794 12795 12801 
##     C     C     C     C     C     C     C     C     C     C     C     D 
## 12804 12805 12816 12824 12825 12826 12829 12832 12836 12842 12845 12848 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 12852 12853 12854 12857 12859 12862 12867 12869 12870 12871 12873 12886 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 12887 12891 12894 12897 12898 12907 12911 12915 12918 12924 12925 12930 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 12933 12939 12941 12945 12946 12947 12952 12958 12960 12965 12966 12967 
##     D     D     D     D     C     D     D     D     D     C     C     C 
## 12968 12970 12971 12977 12979 12981 12982 12986 12992 12993 13007 13009 
##     C     C     D     D     D     D     D     D     D     D     D     D 
## 13011 13012 13015 13021 13027 13031 13033 13039 13040 13041 13042 13049 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13051 13057 13065 13066 13070 13075 13083 13087 13090 13091 13092 13096 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13097 13101 13105 13111 13112 13125 13129 13130 13132 13134 13135 13136 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13142 13144 13150 13155 13156 13161 13164 13165 13170 13171 13177 13180 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13184 13188 13189 13190 13195 13197 13200 13203 13206 13207 13214 13223 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13229 13234 13235 13240 13243 13245 13248 13253 13256 13257 13259 13263 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13270 13276 13279 13281 13287 13290 13292 13294 13295 13299 13301 13302 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13306 13308 13315 13321 13323 13327 13328 13334 13338 13340 13342 13346 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13352 13354 13355 13357 13369 13371 13379 13386 13387 13391 13392 13400 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13404 13409 13418 13421 13429 13430 13431 13433 13434 13436 13443 13446 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13447 13453 13455 13457 13458 13469 13477 13478 13481 13482 13485 13487 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13490 13491 13492 13497 13498 13500 13505 13513 13517 13522 13528 13529 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13530 13536 13541 13542 13544 13545 13546 13547 13550 13557 13559 13564 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13569 13572 13574 13580 13583 13585 13586 13588 13590 13592 13595 13596 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13597 13606 13608 13609 13610 13611 13613 13615 13616 13620 13625 13626 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13628 13634 13638 13641 13646 13648 13651 13652 13654 13655 13666 13671 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13674 13675 13676 13682 13683 13685 13692 13693 13696 13697 13706 13709 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13720 13725 13726 13729 13731 13738 13742 13750 13767 13770 13771 13777 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13778 13782 13784 13787 13788 13791 13793 13794 13795 13796 13808 13815 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13816 13818 13820 13822 13826 13830 13833 13838 13841 13846 13852 13856 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13862 13867 13872 13874 13875 13886 13887 13892 13894 13897 13898 13900 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13902 13908 13915 13916 13917 13923 13935 13937 13951 13955 13962 13966 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13969 13970 13975 13978 13981 13985 13992 13994 13997 14000 14002 14004 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14005 14007 14010 14012 14015 14016 14025 14027 14028 14029 14031 14034 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14035 14040 14044 14050 14051 14054 14062 14063 14064 14065 14073 14076 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14086 14087 14089 14090 14097 14098 14099 14101 14102 14106 14107 14114 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14120 14124 14126 14127 14128 14131 14133 14136 14139 14144 14147 14148 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14150 14152 14159 14165 14169 14175 14182 14186 14187 14192 14194 14196 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14199 14208 14212 14218 14220 14222 14225 14226 14242 14248 14250 14254 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14258 14260 14264 14269 14275 14277 14278 14279 14281 14285 14288 14291 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14293 14294 14296 14300 14303 14305 14313 14315 14319 14322 14326 14328 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14329 14333 14339 14342 14343 14349 14350 14354 14357 14360 14364 14365 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14368 14372 14373 14377 14379 14381 14383 14385 14390 14396 14399 14402 
##     D     C     D     D     D     D     D     D     D     D     D     D 
## 14403 14404 14405 14406 14409 14414 14415 14417 14418 14423 14427 14429 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14432 14435 14438 14439 14441 14444 14446 14465 14470 14472 14473 14475 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14477 14483 14484 14488 14489 14490 14492 14493 14502 14504 14505 14516 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14521 14522 14523 14524 14526 14530 14536 14537 14538 14539 14541 14545 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14550 14553 14554 14557 14558 14560 14562 14563 14566 14567 14570 14573 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14575 14576 14580 14581 14586 14588 14591 14592 14593 14596 14602 14606 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14609 14615 14621 14623 14626 14628 14638 14644 14645 14646 14649 14650 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14656 14657 14659 14663 14667 14672 14676 14677 14679 14680 14681 14686 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14687 14688 14693 14697 14701 14702 14703 14705 14709 14715 14717 14720 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14724 14726 14731 14737 14741 14742 14743 14747 14749 14750 14751 14753 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14756 14757 14758 14759 14760 14761 14763 14764 14767 14768 14773 14779 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14781 14783 14787 14792 14793 14794 14795 14801 14804 14810 14813 14817 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14818 14820 14824 14829 14834 14835 14838 14839 14844 14847 14851 14857 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14862 14874 14876 14878 14881 14883 14886 14888 14889 14891 14894 14900 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14906 14909 14910 14912 14914 14915 14919 14921 14925 14930 14932 14934 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14935 14936 14939 14943 14944 14950 14953 14966 14973 14974 14976 14977 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14979 14981 14982 14989 14992 14997 14998 14999 15002 15005 15014 15015 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15017 15022 15025 15027 15032 15033 15035 15047 15055 15056 15058 15059 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15064 15065 15066 15072 15074 15075 15078 15080 15087 15092 15096 15100 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15101 15109 15111 15114 15115 15116 15117 15127 15131 15133 15136 15142 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15144 15145 15148 15150 15153 15154 15159 15160 15163 15170 15173 15174 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15175 15177 15181 15184 15190 15191 15198 15199 15210 15213 15219 15223 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15224 15225 15229 15235 15237 15242 15243 15244 15245 15251 15253 15254 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15255 15258 15259 15260 15268 15270 15277 15282 15288 15289 15295 15296 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15298 15306 15307 15310 15318 15319 15327 15330 15331 15332 15338 15342 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15343 15344 15352 15357 15362 15364 15365 15367 15368 15372 15378 15380 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15382 15384 15389 15392 15393 15397 15398 15400 15407 15411 15413 15414 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15417 15432 15434 15436 15438 15441 15443 15455 15456 15458 15469 15470 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15471 15473 15474 15475 15483 15490 15492 15499 15500 15507 15508 15509 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15510 15519 15526 15527 15531 15533 15534 15538 15547 15550 15551 15552 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15553 15554 15556 15557 15559 15563 15565 15566 15568 15571 15575 15578 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15585 15586 15587 15589 15591 15592 15595 15598 15599 15600 15602 15605 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15614 15621 15632 15633 15635 15637 15642 15643 15646 15649 15652 15654 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15658 15660 15661 15669 15674 15675 15676 15677 15679 15681 15682 15683 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15685 15689 15690 15692 15695 15696 15699 15700 15705 15706 15714 15715 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15717 15718 15722 15723 15726 15729 15731 15732 15740 15742 15747 15750 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15754 15760 15764 15769 15773 15778 15784 15798 15799 15804 15810 15813 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15814 15816 15819 15822 15824 15825 15826 15832 15835 15841 15845 15847 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15848 15855 15858 15864 15868 15872 15873 15876 15877 15879 15884 15885 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15887 15890 15892 15894 15898 15901 15902 15903 15926 15930 15935 15937 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15938 15939 15941 15946 15949 15950 15951 15953 15955 15956 15960 15961 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15962 15975 15979 15980 15981 15987 15993 15994 15996 15997 16000 16001 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 16006 16012 16015 16020 16024 16027 16031 16039 16040 16049 16050 16051 
##     D     D     D     E     E     E     E     E     E     E     E     E 
## 16055 16061 16065 16066 16073 16076 16078 16082 16085 16087 16092 16094 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16101 16102 16106 16123 16126 16131 16135 16136 16137 16139 16145 16147 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16151 16158 16160 16164 16169 16174 16177 16182 16189 16192 16197 16198 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16210 16212 16215 16220 16223 16224 16226 16227 16228 16231 16232 16233 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16243 16244 16246 16251 16257 16259 16261 16264 16267 16268 16269 16274 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16282 16286 16295 16298 16299 16301 16305 16308 16311 16315 16321 16323 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16330 16331 16335 16336 16337 16344 16345 16348 16349 16350 16356 16358 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16361 16363 16366 16375 16378 16379 16384 16387 16393 16396 16401 16403 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16404 16410 16412 16413 16424 16426 16428 16430 16432 16435 16436 16440 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16446 16448 16457 16467 16468 16475 16477 16478 16480 16482 16486 16488 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16489 16501 16504 16508 16509 16513 16515 16516 16517 16519 16521 16522 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16526 16528 16529 16531 16532 16535 16536 16539 16540 16541 16544 16545 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16547 16555 16563 16574 16575 16576 16577 16579 16586 16588 16593 16594 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16595 16596 16600 16613 16614 16617 16618 16620 16622 16628 16630 16631 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16636 16639 16644 16646 16647 16655 16657 16660 16661 16662 16669 16676 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16685 16687 16691 16692 16694 16700 16701 16708 16710 16721 16722 16724 
##     E     E     E     E     E     E     E     E     E     D     E     E 
## 16727 16730 16731 16733 16735 16738 16739 16745 16746 16751 16753 16755 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16756 16761 16767 16768 16771 16775 16778 16781 16784 16790 16791 16793 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16794 16798 16804 16808 16811 16813 16815 16818 16821 16823 16827 16830 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16833 16837 16840 16841 16842 16843 16846 16850 16861 16863 16864 16871 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16874 16880 16881 16892 16900 16905 16906 16907 16910 16911 16916 16918 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16920 16921 16928 16929 16936 16937 16951 16952 16956 16958 16963 16972 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16973 16977 16979 16980 16982 16984 16985 16986 16988 16993 16994 16998 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17000 17002 17005 17007 17009 17010 17011 17013 17014 17018 17020 17022 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17024 17032 17033 17043 17046 17049 17051 17052 17053 17055 17056 17062 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17067 17068 17073 17078 17092 17094 17101 17102 17109 17110 17111 17114 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17116 17117 17123 17125 17128 17130 17133 17141 17149 17150 17152 17154 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17156 17158 17160 17167 17180 17183 17184 17185 17186 17187 17192 17196 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17200 17210 17213 17214 17216 17218 17220 17225 17232 17234 17235 17237 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17241 17243 17245 17246 17247 17252 17254 17257 17260 17262 17264 17265 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17267 17269 17287 17295 17309 17314 17316 17317 17321 17322 17323 17324 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17326 17332 17335 17336 17338 17339 17344 17345 17348 17349 17356 17358 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17361 17362 17365 17370 17374 17382 17387 17388 17391 17392 17393 17399 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17403 17404 17405 17409 17411 17413 17414 17418 17420 17421 17425 17431 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17438 17444 17448 17451 17453 17455 17457 17458 17473 17476 17480 17482 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17484 17485 17487 17489 17496 17497 17505 17506 17507 17509 17512 17513 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17514 17520 17521 17523 17524 17530 17538 17539 17541 17544 17553 17556 
##     E     E     E     E     E     E     D     E     E     E     E     E 
## 17557 17564 17565 17567 17569 17573 17575 17578 17579 17580 17583 17588 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17589 17600 17602 17604 17605 17612 17615 17617 17621 17625 17627 17637 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17638 17641 17645 17650 17651 17652 17653 17655 17656 17657 17658 17661 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17664 17667 17669 17672 17686 17687 17689 17691 17693 17694 17699 17700 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17702 17705 17707 17715 17720 17730 17738 17742 17751 17753 17755 17759 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17764 17767 17771 17773 17774 17776 17779 17781 17784 17785 17788 17790 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17802 17803 17804 17805 17806 17821 17824 17826 17827 17828 17831 17835 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17840 17841 17843 17844 17846 17849 17852 17855 17856 17863 17867 17868 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17877 17878 17883 17884 17886 17888 17889 17895 17898 17900 17903 17904 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17911 17919 17920 17921 17922 17925 17940 17946 17950 17951 17957 17964 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17970 17973 17974 17978 17983 17995 17997 17998 18000 18004 18011 18012 
##     E     E     E     E     E     E     D     E     E     E     E     E 
## 18013 18016 18021 18022 18025 18026 18034 18039 18040 18049 18051 18052 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18055 18061 18063 18064 18067 18074 18079 18080 18085 18087 18089 18090 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18095 18096 18099 18100 18101 18103 18106 18120 18129 18131 18132 18133 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18136 18140 18141 18143 18147 18148 18149 18152 18155 18160 18163 18167 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18177 18181 18182 18183 18187 18189 18190 18191 18192 18197 18202 18204 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18205 18209 18210 18217 18222 18223 18228 18236 18237 18239 18240 18242 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18243 18244 18245 18247 18248 18251 18255 18258 18261 18267 18268 18269 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18270 18273 18280 18283 18286 18288 18289 18297 18299 18300 18306 18307 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18308 18311 18314 18318 18319 18321 18323 18325 18330 18331 18333 18335 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18338 18340 18342 18343 18349 18356 18358 18361 18363 18364 18365 18366 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18369 18373 18375 18376 18379 18384 18387 18390 18391 18392 18395 18397 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18400 18401 18405 18407 18408 18409 18410 18412 18417 18421 18423 18426 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18428 18429 18430 18432 18433 18434 18442 18448 18453 18463 18464 18468 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18469 18470 18471 18473 18477 18478 18481 18483 18492 18496 18497 18500 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18501 18502 18505 18509 18510 18511 18516 18518 18521 18525 18528 18529 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18530 18531 18535 18537 18538 18539 18544 18545 18547 18553 18556 18558 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18564 18569 18582 18591 18593 18594 18596 18604 18605 18606 18620 18624 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18630 18632 18634 18636 18644 18645 18649 18651 18655 18656 18657 18658 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18662 18665 18666 18668 18681 18683 18685 18691 18697 18703 18705 18706 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18707 18712 18716 18719 18721 18722 18726 18728 18732 18742 18746 18753 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18768 18770 18772 18773 18775 18776 18780 18782 18785 18787 18790 18793 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18802 18803 18804 18806 18808 18809 18812 18813 18814 18815 18816 18823 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18824 18828 18829 18830 18832 18844 18847 18849 18852 18853 18855 18857 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18858 18861 18862 18863 18865 18868 18872 18873 18875 18877 18881 18882 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18883 18884 18889 18893 18895 18899 18913 18914 18916 18917 18919 18923 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18928 18943 18944 18954 18956 18957 18959 18970 18971 18978 18980 18983 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18984 18986 18988 18989 18993 18997 19000 19006 19009 19010 19011 19014 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19016 19025 19037 19038 19040 19044 19046 19047 19051 19053 19054 19056 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19059 19064 19065 19067 19070 19071 19072 19073 19075 19079 19081 19086 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19087 19089 19094 19098 19099 19102 19103 19106 19109 19113 19116 19119 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19121 19123 19124 19132 19137 19141 19145 19149 19151 19154 19156 19157 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19163 19168 19173 19174 19177 19178 19181 19185 19187 19188 19191 19194 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19200 19203 19205 19209 19217 19222 19223 19226 19229 19230 19236 19237 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19243 19246 19251 19254 19258 19261 19264 19278 19280 19282 19287 19289 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19297 19301 19304 19305 19306 19311 19330 19338 19340 19345 19347 19348 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19358 19363 19364 19365 19367 19369 19376 19387 19389 19394 19395 19396 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19402 19405 19407 19410 19413 19414 19418 19423 19424 19425 19430 19435 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19440 19441 19445 19446 19450 19456 19463 19464 19465 19474 19480 19487 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19488 19491 19492 19493 19498 19499 19501 19502 19508 19512 19520 19522 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19523 19526 19529 19532 19533 19540 19541 19542 19546 19548 19550 19551 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19556 19557 19562 19565 19574 19575 19577 19578 19585 19596 19600 19606 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19607 19610 19611 19614 19620 
##     E     E     E     E     E 
## Levels: A B C D E
```
