####Practical Machine Learning - Human Activity Recognition Prediction Project#####
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
##         OOB estimate of  error rate: 0.6%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3902    3    1    0    0 0.001024066
## B   16 2635    7    0    0 0.008653123
## C    0   15 2379    2    0 0.007095159
## D    0    0   23 2226    3 0.011545293
## E    0    0    2   10 2513 0.004752475
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
##          A 1672    2    0    0    0
##          B    5 1133    1    0    0
##          C    0    5 1017    4    0
##          D    0    0   19  945    0
##          E    0    0    0    4 1078
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9932          
##                  95% CI : (0.9908, 0.9951)
##     No Information Rate : 0.285           
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9914          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9970   0.9939   0.9807   0.9916   1.0000
## Specificity            0.9995   0.9987   0.9981   0.9961   0.9992
## Pos Pred Value         0.9988   0.9947   0.9912   0.9803   0.9963
## Neg Pred Value         0.9988   0.9985   0.9959   0.9984   1.0000
## Prevalence             0.2850   0.1937   0.1762   0.1619   0.1832
## Detection Rate         0.2841   0.1925   0.1728   0.1606   0.1832
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9983   0.9963   0.9894   0.9939   0.9996
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
##  $ roll_belt           : num  1.41 1.42 1.48 1.42 1.43 1.45 1.45 1.43 1.42 1.45 ...
##  $ pitch_belt          : num  8.07 8.07 8.07 8.09 8.16 8.17 8.18 8.18 8.21 8.2 ...
##  $ yaw_belt            : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt    : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ gyros_belt_x        : num  0 0 0.02 0.02 0.02 0.03 0.03 0.02 0.02 0 ...
##  $ gyros_belt_y        : num  0 0 0.02 0 0 0 0 0 0 0 ...
##  $ gyros_belt_z        : num  -0.02 -0.02 -0.02 -0.02 -0.02 0 -0.02 -0.02 -0.02 0 ...
##  $ accel_belt_x        : int  -21 -20 -21 -22 -20 -21 -21 -22 -22 -21 ...
##  $ accel_belt_y        : int  4 5 2 3 2 4 2 2 4 2 ...
##  $ accel_belt_z        : int  22 23 24 21 24 22 23 23 21 22 ...
##  $ magnet_belt_x       : int  -3 -2 -6 -4 1 -3 -5 -2 -8 -1 ...
##  $ magnet_belt_y       : int  599 600 600 599 602 609 596 602 598 597 ...
##  $ magnet_belt_z       : int  -313 -305 -302 -311 -312 -308 -317 -319 -310 -310 ...
##  $ roll_arm            : num  -128 -128 -128 -128 -128 -128 -128 -128 -128 -129 ...
##  $ pitch_arm           : num  22.5 22.5 22.1 21.9 21.7 21.6 21.5 21.5 21.4 21.4 ...
##  $ yaw_arm             : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
##  $ total_accel_arm     : int  34 34 34 34 34 34 34 34 34 34 ...
##  $ gyros_arm_x         : num  0 0.02 0 0 0.02 0.02 0.02 0.02 0.02 0.02 ...
##  $ gyros_arm_y         : num  0 -0.02 -0.03 -0.03 -0.03 -0.03 -0.03 -0.03 0 0 ...
##  $ gyros_arm_z         : num  -0.02 -0.02 0 0 -0.02 -0.02 0 0 -0.03 -0.03 ...
##  $ accel_arm_x         : int  -288 -289 -289 -289 -288 -288 -290 -288 -288 -289 ...
##  $ accel_arm_y         : int  109 110 111 111 109 110 110 111 111 111 ...
##  $ accel_arm_z         : int  -123 -126 -123 -125 -122 -124 -123 -123 -124 -124 ...
##  $ magnet_arm_x        : int  -368 -368 -374 -373 -369 -376 -366 -363 -371 -374 ...
##  $ magnet_arm_y        : int  337 344 337 336 341 334 339 343 331 342 ...
##  $ magnet_arm_z        : int  516 513 506 509 518 516 509 520 523 510 ...
##  $ roll_dumbbell       : num  13.1 12.9 13.4 13.1 13.2 ...
##  $ pitch_dumbbell      : num  -70.5 -70.3 -70.4 -70.2 -70.4 ...
##  $ yaw_dumbbell        : num  -84.9 -85.1 -84.9 -85.1 -84.9 ...
##  $ total_accel_dumbbell: int  37 37 37 37 37 37 37 37 37 37 ...
##  $ gyros_dumbbell_x    : num  0 0 0 0 0 0 0 0 0.02 0 ...
##  $ gyros_dumbbell_y    : num  -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 ...
##  $ gyros_dumbbell_z    : num  0 0 0 0 0 0 0 0 -0.02 0 ...
##  $ accel_dumbbell_x    : int  -234 -232 -233 -232 -232 -235 -233 -233 -234 -234 ...
##  $ accel_dumbbell_y    : int  47 46 48 47 47 48 47 47 48 47 ...
##  $ accel_dumbbell_z    : int  -271 -270 -270 -270 -269 -270 -269 -270 -268 -270 ...
##  $ magnet_dumbbell_x   : int  -559 -561 -554 -551 -549 -558 -564 -554 -554 -554 ...
##  $ magnet_dumbbell_y   : int  293 298 292 295 292 291 299 291 295 294 ...
##  $ magnet_dumbbell_z   : num  -65 -63 -68 -70 -65 -69 -64 -65 -68 -63 ...
##  $ roll_forearm        : num  28.4 28.3 28 27.9 27.7 27.7 27.6 27.5 27.2 27.2 ...
##  $ pitch_forearm       : num  -63.9 -63.9 -63.9 -63.9 -63.8 -63.8 -63.8 -63.8 -63.9 -63.9 ...
##  $ yaw_forearm         : num  -153 -152 -152 -152 -152 -152 -152 -152 -151 -151 ...
##  $ total_accel_forearm : int  36 36 36 36 36 36 36 36 36 36 ...
##  $ gyros_forearm_x     : num  0.03 0.03 0.02 0.02 0.03 0.02 0.02 0.02 0 0 ...
##  $ gyros_forearm_y     : num  0 -0.02 0 0 0 0 -0.02 0.02 -0.02 -0.02 ...
##  $ gyros_forearm_z     : num  -0.02 0 -0.02 -0.02 -0.02 -0.02 -0.02 -0.03 -0.03 -0.02 ...
##  $ accel_forearm_x     : int  192 196 189 195 193 190 193 191 193 192 ...
##  $ accel_forearm_y     : int  203 204 206 205 204 205 205 203 202 201 ...
##  $ accel_forearm_z     : int  -215 -213 -214 -215 -214 -215 -214 -215 -214 -214 ...
##  $ magnet_forearm_x    : int  -17 -18 -17 -18 -16 -22 -17 -11 -14 -16 ...
##  $ magnet_forearm_y    : num  654 658 655 659 653 656 657 657 659 656 ...
##  $ magnet_forearm_z    : num  476 469 473 470 476 473 465 478 478 472 ...
##  $ classe              : Factor w/ 5 levels "A","B","C","D",..: 1 1 1 1 1 1 1 1 1 1 ...
```

```r
head(training)
```

```
##    roll_belt pitch_belt yaw_belt total_accel_belt gyros_belt_x
## 1       1.41       8.07    -94.4                3         0.00
## 3       1.42       8.07    -94.4                3         0.00
## 5       1.48       8.07    -94.4                3         0.02
## 7       1.42       8.09    -94.4                3         0.02
## 9       1.43       8.16    -94.4                3         0.02
## 10      1.45       8.17    -94.4                3         0.03
##    gyros_belt_y gyros_belt_z accel_belt_x accel_belt_y accel_belt_z
## 1          0.00        -0.02          -21            4           22
## 3          0.00        -0.02          -20            5           23
## 5          0.02        -0.02          -21            2           24
## 7          0.00        -0.02          -22            3           21
## 9          0.00        -0.02          -20            2           24
## 10         0.00         0.00          -21            4           22
##    magnet_belt_x magnet_belt_y magnet_belt_z roll_arm pitch_arm yaw_arm
## 1             -3           599          -313     -128      22.5    -161
## 3             -2           600          -305     -128      22.5    -161
## 5             -6           600          -302     -128      22.1    -161
## 7             -4           599          -311     -128      21.9    -161
## 9              1           602          -312     -128      21.7    -161
## 10            -3           609          -308     -128      21.6    -161
##    total_accel_arm gyros_arm_x gyros_arm_y gyros_arm_z accel_arm_x
## 1               34        0.00        0.00       -0.02        -288
## 3               34        0.02       -0.02       -0.02        -289
## 5               34        0.00       -0.03        0.00        -289
## 7               34        0.00       -0.03        0.00        -289
## 9               34        0.02       -0.03       -0.02        -288
## 10              34        0.02       -0.03       -0.02        -288
##    accel_arm_y accel_arm_z magnet_arm_x magnet_arm_y magnet_arm_z
## 1          109        -123         -368          337          516
## 3          110        -126         -368          344          513
## 5          111        -123         -374          337          506
## 7          111        -125         -373          336          509
## 9          109        -122         -369          341          518
## 10         110        -124         -376          334          516
##    roll_dumbbell pitch_dumbbell yaw_dumbbell total_accel_dumbbell
## 1       13.05217      -70.49400    -84.87394                   37
## 3       12.85075      -70.27812    -85.14078                   37
## 5       13.37872      -70.42856    -84.85306                   37
## 7       13.12695      -70.24757    -85.09961                   37
## 9       13.15463      -70.42520    -84.91563                   37
## 10      13.33034      -70.85059    -84.44602                   37
##    gyros_dumbbell_x gyros_dumbbell_y gyros_dumbbell_z accel_dumbbell_x
## 1                 0            -0.02                0             -234
## 3                 0            -0.02                0             -232
## 5                 0            -0.02                0             -233
## 7                 0            -0.02                0             -232
## 9                 0            -0.02                0             -232
## 10                0            -0.02                0             -235
##    accel_dumbbell_y accel_dumbbell_z magnet_dumbbell_x magnet_dumbbell_y
## 1                47             -271              -559               293
## 3                46             -270              -561               298
## 5                48             -270              -554               292
## 7                47             -270              -551               295
## 9                47             -269              -549               292
## 10               48             -270              -558               291
##    magnet_dumbbell_z roll_forearm pitch_forearm yaw_forearm
## 1                -65         28.4         -63.9        -153
## 3                -63         28.3         -63.9        -152
## 5                -68         28.0         -63.9        -152
## 7                -70         27.9         -63.9        -152
## 9                -65         27.7         -63.8        -152
## 10               -69         27.7         -63.8        -152
##    total_accel_forearm gyros_forearm_x gyros_forearm_y gyros_forearm_z
## 1                   36            0.03            0.00           -0.02
## 3                   36            0.03           -0.02            0.00
## 5                   36            0.02            0.00           -0.02
## 7                   36            0.02            0.00           -0.02
## 9                   36            0.03            0.00           -0.02
## 10                  36            0.02            0.00           -0.02
##    accel_forearm_x accel_forearm_y accel_forearm_z magnet_forearm_x
## 1              192             203            -215              -17
## 3              196             204            -213              -18
## 5              189             206            -214              -17
## 7              195             205            -215              -18
## 9              193             204            -214              -16
## 10             190             205            -215              -22
##    magnet_forearm_y magnet_forearm_z classe
## 1               654              476      A
## 3               658              469      A
## 5               655              473      A
## 7               659              470      A
## 9               653              476      A
## 10              656              473      A
```

```r
summary(training)
```

```
##    roll_belt        pitch_belt          yaw_belt      total_accel_belt
##  Min.   :-28.90   Min.   :-54.9000   Min.   :-180.0   Min.   : 0.00   
##  1st Qu.:  1.09   1st Qu.:  1.6600   1st Qu.: -88.3   1st Qu.: 3.00   
##  Median :113.00   Median :  5.2600   Median : -12.8   Median :17.00   
##  Mean   : 64.45   Mean   :  0.2506   Mean   : -11.0   Mean   :11.33   
##  3rd Qu.:123.00   3rd Qu.: 15.0000   3rd Qu.:  13.1   3rd Qu.:18.00   
##  Max.   :162.00   Max.   : 60.3000   Max.   : 179.0   Max.   :29.00   
##   gyros_belt_x        gyros_belt_y       gyros_belt_z    
##  Min.   :-1.040000   Min.   :-0.64000   Min.   :-1.4600  
##  1st Qu.:-0.030000   1st Qu.: 0.00000   1st Qu.:-0.2000  
##  Median : 0.030000   Median : 0.02000   Median :-0.1000  
##  Mean   :-0.006036   Mean   : 0.03902   Mean   :-0.1323  
##  3rd Qu.: 0.110000   3rd Qu.: 0.11000   3rd Qu.:-0.0200  
##  Max.   : 2.200000   Max.   : 0.64000   Max.   : 1.6200  
##   accel_belt_x       accel_belt_y     accel_belt_z     magnet_belt_x   
##  Min.   :-120.000   Min.   :-69.00   Min.   :-275.00   Min.   :-49.00  
##  1st Qu.: -21.000   1st Qu.:  3.00   1st Qu.:-162.00   1st Qu.:  9.00  
##  Median : -15.000   Median : 35.00   Median :-153.00   Median : 35.00  
##  Mean   :  -5.534   Mean   : 30.15   Mean   : -72.72   Mean   : 55.79  
##  3rd Qu.:  -5.000   3rd Qu.: 61.00   3rd Qu.:  28.00   3rd Qu.: 60.00  
##  Max.   :  85.000   Max.   :164.00   Max.   : 105.00   Max.   :481.00  
##  magnet_belt_y   magnet_belt_z       roll_arm         pitch_arm     
##  Min.   :360.0   Min.   :-621.0   Min.   :-180.00   Min.   :-88.80  
##  1st Qu.:581.0   1st Qu.:-375.0   1st Qu.: -31.50   1st Qu.:-25.70  
##  Median :601.0   Median :-320.0   Median :   0.00   Median :  0.00  
##  Mean   :593.5   Mean   :-345.8   Mean   :  17.58   Mean   : -4.38  
##  3rd Qu.:610.0   3rd Qu.:-306.0   3rd Qu.:  77.10   3rd Qu.: 11.50  
##  Max.   :673.0   Max.   : 293.0   Max.   : 180.00   Max.   : 87.10  
##     yaw_arm          total_accel_arm  gyros_arm_x        gyros_arm_y     
##  Min.   :-180.0000   Min.   : 1.00   Min.   :-6.37000   Min.   :-3.4400  
##  1st Qu.: -43.3000   1st Qu.:17.00   1st Qu.:-1.36000   1st Qu.:-0.7900  
##  Median :   0.0000   Median :27.00   Median : 0.06000   Median :-0.2200  
##  Mean   :  -0.5888   Mean   :25.46   Mean   : 0.02991   Mean   :-0.2503  
##  3rd Qu.:  45.9000   3rd Qu.:33.00   3rd Qu.: 1.54000   3rd Qu.: 0.1600  
##  Max.   : 180.0000   Max.   :66.00   Max.   : 4.87000   Max.   : 2.8400  
##   gyros_arm_z       accel_arm_x       accel_arm_y       accel_arm_z     
##  Min.   :-2.2800   Min.   :-404.00   Min.   :-318.00   Min.   :-636.00  
##  1st Qu.:-0.0800   1st Qu.:-241.00   1st Qu.: -54.00   1st Qu.:-141.00  
##  Median : 0.2300   Median : -44.00   Median :  14.00   Median : -46.00  
##  Mean   : 0.2655   Mean   : -60.66   Mean   :  32.87   Mean   : -70.65  
##  3rd Qu.: 0.7200   3rd Qu.:  82.00   3rd Qu.: 139.00   3rd Qu.:  24.00  
##  Max.   : 2.9900   Max.   : 437.00   Max.   : 308.00   Max.   : 271.00  
##   magnet_arm_x     magnet_arm_y     magnet_arm_z  roll_dumbbell    
##  Min.   :-584.0   Min.   :-392.0   Min.   :-597   Min.   :-153.71  
##  1st Qu.:-300.0   1st Qu.:  -6.0   1st Qu.: 137   1st Qu.: -17.50  
##  Median : 281.0   Median : 203.0   Median : 445   Median :  48.32  
##  Mean   : 189.2   Mean   : 157.5   Mean   : 308   Mean   :  24.25  
##  3rd Qu.: 636.0   3rd Qu.: 324.0   3rd Qu.: 546   3rd Qu.:  67.93  
##  Max.   : 782.0   Max.   : 583.0   Max.   : 687   Max.   : 153.55  
##  pitch_dumbbell     yaw_dumbbell      total_accel_dumbbell
##  Min.   :-149.59   Min.   :-148.766   Min.   : 0.00       
##  1st Qu.: -41.19   1st Qu.: -77.713   1st Qu.: 4.00       
##  Median : -20.90   Median :  -4.072   Median :11.00       
##  Mean   : -10.92   Mean   :   1.718   Mean   :13.77       
##  3rd Qu.:  17.14   3rd Qu.:  79.984   3rd Qu.:20.00       
##  Max.   : 149.40   Max.   : 154.754   Max.   :58.00       
##  gyros_dumbbell_x    gyros_dumbbell_y   gyros_dumbbell_z  
##  Min.   :-204.0000   Min.   :-2.10000   Min.   : -2.3800  
##  1st Qu.:  -0.0300   1st Qu.:-0.14000   1st Qu.: -0.3100  
##  Median :   0.1400   Median : 0.03000   Median : -0.1300  
##  Mean   :   0.1584   Mean   : 0.04825   Mean   : -0.1231  
##  3rd Qu.:   0.3500   3rd Qu.: 0.21000   3rd Qu.:  0.0300  
##  Max.   :   2.1700   Max.   :52.00000   Max.   :317.0000  
##  accel_dumbbell_x  accel_dumbbell_y  accel_dumbbell_z  magnet_dumbbell_x
##  Min.   :-419.00   Min.   :-189.00   Min.   :-319.00   Min.   :-639.0   
##  1st Qu.: -51.00   1st Qu.:  -8.00   1st Qu.:-142.00   1st Qu.:-535.0   
##  Median :  -9.00   Median :  43.00   Median :  -1.00   Median :-480.0   
##  Mean   : -28.92   Mean   :  53.09   Mean   : -38.71   Mean   :-327.1   
##  3rd Qu.:  11.00   3rd Qu.: 112.00   3rd Qu.:  38.00   3rd Qu.:-303.0   
##  Max.   : 235.00   Max.   : 315.00   Max.   : 318.00   Max.   : 592.0   
##  magnet_dumbbell_y magnet_dumbbell_z  roll_forearm     pitch_forearm   
##  Min.   :-3600.0   Min.   :-262.00   Min.   :-180.00   Min.   :-72.50  
##  1st Qu.:  231.0   1st Qu.: -45.00   1st Qu.:  -0.46   1st Qu.:  0.00  
##  Median :  311.0   Median :  14.00   Median :  22.10   Median :  9.37  
##  Mean   :  219.6   Mean   :  46.24   Mean   :  34.16   Mean   : 10.76  
##  3rd Qu.:  389.0   3rd Qu.:  96.00   3rd Qu.: 140.00   3rd Qu.: 28.40  
##  Max.   :  633.0   Max.   : 451.00   Max.   : 180.00   Max.   : 89.80  
##   yaw_forearm      total_accel_forearm gyros_forearm_x   
##  Min.   :-180.00   Min.   :  0.0       Min.   :-22.0000  
##  1st Qu.: -67.80   1st Qu.: 29.0       1st Qu.: -0.2200  
##  Median :   0.00   Median : 36.0       Median :  0.0500  
##  Mean   :  19.71   Mean   : 34.8       Mean   :  0.1588  
##  3rd Qu.: 110.00   3rd Qu.: 41.0       3rd Qu.:  0.5600  
##  Max.   : 180.00   Max.   :108.0       Max.   :  3.9700  
##  gyros_forearm_y     gyros_forearm_z    accel_forearm_x   accel_forearm_y
##  Min.   : -7.02000   Min.   : -8.0900   Min.   :-498.00   Min.   :-585   
##  1st Qu.: -1.48000   1st Qu.: -0.2000   1st Qu.:-178.00   1st Qu.:  60   
##  Median :  0.02000   Median :  0.0800   Median : -57.00   Median : 202   
##  Mean   :  0.05994   Mean   :  0.1488   Mean   : -61.74   Mean   : 166   
##  3rd Qu.:  1.61000   3rd Qu.:  0.4800   3rd Qu.:  76.00   3rd Qu.: 314   
##  Max.   :311.00000   Max.   :231.0000   Max.   : 477.00   Max.   : 923   
##  accel_forearm_z   magnet_forearm_x  magnet_forearm_y magnet_forearm_z
##  Min.   :-410.00   Min.   :-1280.0   Min.   :-896.0   Min.   :-973.0  
##  1st Qu.:-181.00   1st Qu.: -619.0   1st Qu.:  17.0   1st Qu.: 194.0  
##  Median : -40.00   Median : -385.0   Median : 595.0   Median : 514.0  
##  Mean   : -55.45   Mean   : -316.4   Mean   : 385.3   Mean   : 394.8  
##  3rd Qu.:  26.00   3rd Qu.:  -81.0   3rd Qu.: 739.0   3rd Qu.: 654.0  
##  Max.   : 291.00   Max.   :  666.0   Max.   :1480.0   Max.   :1090.0  
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
##  $ roll_belt           : num  1.41 1.48 1.45 1.42 1.42 1.55 1.57 1.52 1.52 1.53 ...
##  $ pitch_belt          : num  8.07 8.05 8.06 8.13 8.2 8.08 8.09 8.16 8.17 8.17 ...
##  $ yaw_belt            : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt    : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ gyros_belt_x        : num  0.02 0.02 0.02 0.02 0.02 0 0.02 0.03 0.03 0.02 ...
##  $ gyros_belt_y        : num  0 0 0 0 0 0.02 0.02 0 0 0.02 ...
##  $ gyros_belt_z        : num  -0.02 -0.03 -0.02 -0.02 0 0 -0.02 -0.02 -0.02 -0.02 ...
##  $ accel_belt_x        : int  -22 -22 -21 -22 -22 -21 -21 -20 -21 -21 ...
##  $ accel_belt_y        : int  4 3 4 4 4 5 3 4 4 4 ...
##  $ accel_belt_z        : int  22 21 21 21 21 21 21 23 21 22 ...
##  $ magnet_belt_x       : int  -7 -6 0 -2 -3 1 -2 -4 2 0 ...
##  $ magnet_belt_y       : int  608 604 603 603 606 600 604 606 593 601 ...
##  $ magnet_belt_z       : int  -311 -310 -312 -313 -309 -316 -313 -320 -308 -308 ...
##  $ roll_arm            : num  -128 -128 -128 -128 -128 -129 -129 -129 -129 -129 ...
##  $ pitch_arm           : num  22.5 22.1 22 21.8 21.4 21.2 20.8 20.7 20.7 20.7 ...
##  $ yaw_arm             : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
##  $ total_accel_arm     : int  34 34 34 34 34 34 34 34 34 34 ...
##  $ gyros_arm_x         : num  0.02 0.02 0.02 0.02 0.02 0.02 0.03 -0.02 0 0 ...
##  $ gyros_arm_y         : num  -0.02 -0.03 -0.03 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 0 ...
##  $ gyros_arm_z         : num  -0.02 0.02 0 0 -0.02 -0.03 -0.02 0 0 0 ...
##  $ accel_arm_x         : int  -290 -289 -289 -289 -287 -288 -289 -290 -289 -290 ...
##  $ accel_arm_y         : int  110 111 111 111 111 108 111 109 109 109 ...
##  $ accel_arm_z         : int  -125 -123 -122 -124 -124 -124 -123 -125 -125 -126 ...
##  $ magnet_arm_x        : int  -369 -372 -369 -372 -372 -373 -372 -367 -366 -371 ...
##  $ magnet_arm_y        : int  337 344 342 338 338 336 338 337 349 331 ...
##  $ magnet_arm_z        : int  513 512 513 510 509 510 510 514 523 512 ...
##  $ roll_dumbbell       : num  13.1 13.4 13.4 12.8 13.4 ...
##  $ pitch_dumbbell      : num  -70.6 -70.4 -70.8 -70.3 -70.8 ...
##  $ yaw_dumbbell        : num  -84.7 -84.9 -84.5 -85.1 -84.5 ...
##  $ total_accel_dumbbell: int  37 37 37 37 37 36 37 37 37 37 ...
##  $ gyros_dumbbell_x    : num  0 0 0 0 0 0.02 0 0 0 0 ...
##  $ gyros_dumbbell_y    : num  -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 ...
##  $ gyros_dumbbell_z    : num  0 -0.02 0 0 -0.02 -0.02 0 -0.02 -0.02 -0.02 ...
##  $ accel_dumbbell_x    : int  -233 -232 -234 -234 -234 -231 -233 -234 -234 -233 ...
##  $ accel_dumbbell_y    : int  47 48 48 46 48 47 48 47 47 45 ...
##  $ accel_dumbbell_z    : int  -269 -269 -269 -272 -269 -268 -270 -271 -270 -271 ...
##  $ magnet_dumbbell_x   : int  -555 -552 -558 -555 -552 -557 -554 -552 -562 -558 ...
##  $ magnet_dumbbell_y   : int  296 303 294 300 302 292 301 291 298 294 ...
##  $ magnet_dumbbell_z   : num  -64 -60 -66 -74 -69 -62 -65 -60 -64 -64 ...
##  $ roll_forearm        : num  28.3 28.1 27.9 27.8 27.2 27 27 26.8 26.8 26.8 ...
##  $ pitch_forearm       : num  -63.9 -63.9 -63.9 -63.8 -63.9 -64 -63.9 -63.6 -63.6 -63.6 ...
##  $ yaw_forearm         : num  -153 -152 -152 -152 -151 -151 -151 -151 -151 -151 ...
##  $ total_accel_forearm : int  36 36 36 36 36 36 36 36 36 36 ...
##  $ gyros_forearm_x     : num  0.02 0.02 0.02 0.02 0 0.02 0.02 0.02 0.02 0.02 ...
##  $ gyros_forearm_y     : num  0 -0.02 -0.02 -0.02 0 0 -0.03 -0.02 -0.02 0 ...
##  $ gyros_forearm_z     : num  -0.02 0 -0.03 0 -0.03 -0.02 -0.02 -0.03 -0.03 -0.02 ...
##  $ accel_forearm_x     : int  192 189 193 193 193 192 191 195 189 192 ...
##  $ accel_forearm_y     : int  203 206 203 205 205 206 206 205 204 203 ...
##  $ accel_forearm_z     : int  -216 -214 -215 -213 -215 -216 -213 -217 -217 -217 ...
##  $ magnet_forearm_x    : int  -18 -16 -9 -9 -15 -16 -17 -12 -4 -13 ...
##  $ magnet_forearm_y    : num  661 658 660 660 655 653 654 657 661 660 ...
##  $ magnet_forearm_z    : num  473 469 478 474 472 472 478 469 479 469 ...
##  $ classe              : Factor w/ 5 levels "A","B","C","D",..: 1 1 1 1 1 1 1 1 1 1 ...
```

```r
head(validation)
```

```
##    roll_belt pitch_belt yaw_belt total_accel_belt gyros_belt_x
## 2       1.41       8.07    -94.4                3         0.02
## 4       1.48       8.05    -94.4                3         0.02
## 6       1.45       8.06    -94.4                3         0.02
## 8       1.42       8.13    -94.4                3         0.02
## 13      1.42       8.20    -94.4                3         0.02
## 18      1.55       8.08    -94.4                3         0.00
##    gyros_belt_y gyros_belt_z accel_belt_x accel_belt_y accel_belt_z
## 2          0.00        -0.02          -22            4           22
## 4          0.00        -0.03          -22            3           21
## 6          0.00        -0.02          -21            4           21
## 8          0.00        -0.02          -22            4           21
## 13         0.00         0.00          -22            4           21
## 18         0.02         0.00          -21            5           21
##    magnet_belt_x magnet_belt_y magnet_belt_z roll_arm pitch_arm yaw_arm
## 2             -7           608          -311     -128      22.5    -161
## 4             -6           604          -310     -128      22.1    -161
## 6              0           603          -312     -128      22.0    -161
## 8             -2           603          -313     -128      21.8    -161
## 13            -3           606          -309     -128      21.4    -161
## 18             1           600          -316     -129      21.2    -161
##    total_accel_arm gyros_arm_x gyros_arm_y gyros_arm_z accel_arm_x
## 2               34        0.02       -0.02       -0.02        -290
## 4               34        0.02       -0.03        0.02        -289
## 6               34        0.02       -0.03        0.00        -289
## 8               34        0.02       -0.02        0.00        -289
## 13              34        0.02       -0.02       -0.02        -287
## 18              34        0.02       -0.02       -0.03        -288
##    accel_arm_y accel_arm_z magnet_arm_x magnet_arm_y magnet_arm_z
## 2          110        -125         -369          337          513
## 4          111        -123         -372          344          512
## 6          111        -122         -369          342          513
## 8          111        -124         -372          338          510
## 13         111        -124         -372          338          509
## 18         108        -124         -373          336          510
##    roll_dumbbell pitch_dumbbell yaw_dumbbell total_accel_dumbbell
## 2       13.13074      -70.63751    -84.71065                   37
## 4       13.43120      -70.39379    -84.87363                   37
## 6       13.38246      -70.81759    -84.46500                   37
## 8       12.75083      -70.34768    -85.09708                   37
## 13      13.38246      -70.81759    -84.46500                   37
## 18      13.20646      -70.39037    -84.93667                   36
##    gyros_dumbbell_x gyros_dumbbell_y gyros_dumbbell_z accel_dumbbell_x
## 2              0.00            -0.02             0.00             -233
## 4              0.00            -0.02            -0.02             -232
## 6              0.00            -0.02             0.00             -234
## 8              0.00            -0.02             0.00             -234
## 13             0.00            -0.02            -0.02             -234
## 18             0.02            -0.02            -0.02             -231
##    accel_dumbbell_y accel_dumbbell_z magnet_dumbbell_x magnet_dumbbell_y
## 2                47             -269              -555               296
## 4                48             -269              -552               303
## 6                48             -269              -558               294
## 8                46             -272              -555               300
## 13               48             -269              -552               302
## 18               47             -268              -557               292
##    magnet_dumbbell_z roll_forearm pitch_forearm yaw_forearm
## 2                -64         28.3         -63.9        -153
## 4                -60         28.1         -63.9        -152
## 6                -66         27.9         -63.9        -152
## 8                -74         27.8         -63.8        -152
## 13               -69         27.2         -63.9        -151
## 18               -62         27.0         -64.0        -151
##    total_accel_forearm gyros_forearm_x gyros_forearm_y gyros_forearm_z
## 2                   36            0.02            0.00           -0.02
## 4                   36            0.02           -0.02            0.00
## 6                   36            0.02           -0.02           -0.03
## 8                   36            0.02           -0.02            0.00
## 13                  36            0.00            0.00           -0.03
## 18                  36            0.02            0.00           -0.02
##    accel_forearm_x accel_forearm_y accel_forearm_z magnet_forearm_x
## 2              192             203            -216              -18
## 4              189             206            -214              -16
## 6              193             203            -215               -9
## 8              193             205            -213               -9
## 13             193             205            -215              -15
## 18             192             206            -216              -16
##    magnet_forearm_y magnet_forearm_z classe
## 2               661              473      A
## 4               658              469      A
## 6               660              478      A
## 8               660              474      A
## 13              655              472      A
## 18              653              472      A
```

```r
summary(validation)
```

```
##    roll_belt       pitch_belt          yaw_belt       total_accel_belt
##  Min.   :-28.8   Min.   :-55.8000   Min.   :-179.00   Min.   : 0.00   
##  1st Qu.:  1.1   1st Qu.:  1.9600   1st Qu.: -88.30   1st Qu.: 3.00   
##  Median :113.0   Median :  5.3400   Median : -13.50   Median :17.00   
##  Mean   : 64.3   Mean   :  0.4329   Mean   : -11.69   Mean   :11.27   
##  3rd Qu.:123.0   3rd Qu.: 14.8000   3rd Qu.:  10.90   3rd Qu.:18.00   
##  Max.   :161.0   Max.   : 60.2000   Max.   : 179.00   Max.   :27.00   
##   gyros_belt_x        gyros_belt_y      gyros_belt_z      accel_belt_x    
##  Min.   :-1.000000   Min.   :-0.5300   Min.   :-1.2800   Min.   :-81.000  
##  1st Qu.:-0.030000   1st Qu.: 0.0000   1st Qu.:-0.1800   1st Qu.:-21.000  
##  Median : 0.030000   Median : 0.0200   Median :-0.1000   Median :-15.000  
##  Mean   :-0.004556   Mean   : 0.0409   Mean   :-0.1265   Mean   : -5.737  
##  3rd Qu.: 0.110000   3rd Qu.: 0.1100   3rd Qu.: 0.0000   3rd Qu.: -5.000  
##  Max.   : 2.220000   Max.   : 0.5100   Max.   : 1.6100   Max.   : 83.000  
##   accel_belt_y     accel_belt_z    magnet_belt_x    magnet_belt_y  
##  Min.   :-38.00   Min.   :-262.0   Min.   :-52.00   Min.   :354.0  
##  1st Qu.:  3.00   1st Qu.:-162.0   1st Qu.:  9.00   1st Qu.:582.0  
##  Median : 34.00   Median :-151.0   Median : 34.00   Median :601.0  
##  Mean   : 30.17   Mean   : -72.3   Mean   : 55.15   Mean   :594.1  
##  3rd Qu.: 61.00   3rd Qu.:  27.0   3rd Qu.: 58.00   3rd Qu.:610.0  
##  Max.   :149.00   Max.   : 104.0   Max.   :485.00   Max.   :668.0  
##  magnet_belt_z       roll_arm         pitch_arm          yaw_arm         
##  Min.   :-623.0   Min.   :-180.00   Min.   :-87.900   Min.   :-179.0000  
##  1st Qu.:-374.0   1st Qu.: -32.10   1st Qu.:-26.300   1st Qu.: -42.6000  
##  Median :-319.0   Median :   0.00   Median :  0.000   Median :   0.0000  
##  Mean   :-344.8   Mean   :  18.41   Mean   : -5.154   Mean   :  -0.6886  
##  3rd Qu.:-306.0   3rd Qu.:  77.70   3rd Qu.: 10.600   3rd Qu.:  45.8000  
##  Max.   : 284.0   Max.   : 179.00   Max.   : 88.500   Max.   : 180.0000  
##  total_accel_arm  gyros_arm_x       gyros_arm_y      gyros_arm_z     
##  Min.   : 1.00   Min.   :-6.3700   Min.   :-3.320   Min.   :-2.3300  
##  1st Qu.:17.00   1st Qu.:-1.2400   1st Qu.:-0.820   1st Qu.:-0.0700  
##  Median :27.00   Median : 0.1000   Median :-0.260   Median : 0.2500  
##  Mean   :25.62   Mean   : 0.0728   Mean   :-0.273   Mean   : 0.2788  
##  3rd Qu.:33.00   3rd Qu.: 1.6100   3rd Qu.: 0.130   3rd Qu.: 0.7400  
##  Max.   :65.00   Max.   : 4.8700   Max.   : 2.780   Max.   : 3.0200  
##   accel_arm_x       accel_arm_y       accel_arm_z       magnet_arm_x   
##  Min.   :-377.00   Min.   :-279.00   Min.   :-630.00   Min.   :-575.0  
##  1st Qu.:-244.00   1st Qu.: -55.00   1st Qu.:-147.00   1st Qu.:-299.0  
##  Median : -43.00   Median :  14.00   Median : -48.00   Median : 300.0  
##  Mean   : -59.26   Mean   :  31.96   Mean   : -72.63   Mean   : 197.6  
##  3rd Qu.:  87.00   3rd Qu.: 139.00   3rd Qu.:  22.00   3rd Qu.: 640.0  
##  Max.   : 435.00   Max.   : 303.00   Max.   : 292.00   Max.   : 780.0  
##   magnet_arm_y     magnet_arm_z    roll_dumbbell     pitch_dumbbell   
##  Min.   :-381.0   Min.   :-596.0   Min.   :-152.83   Min.   :-148.50  
##  1st Qu.: -15.0   1st Qu.: 119.0   1st Qu.: -20.38   1st Qu.: -39.87  
##  Median : 200.0   Median : 441.0   Median :  47.89   Median : -21.11  
##  Mean   : 154.5   Mean   : 303.1   Mean   :  22.88   Mean   : -10.45  
##  3rd Qu.: 321.0   3rd Qu.: 543.0   3rd Qu.:  67.29   3rd Qu.:  17.91  
##  Max.   : 582.0   Max.   : 694.0   Max.   : 153.38   Max.   : 129.52  
##   yaw_dumbbell      total_accel_dumbbell gyros_dumbbell_x 
##  Min.   :-150.871   Min.   : 0.00        Min.   :-1.9900  
##  1st Qu.: -77.501   1st Qu.: 4.00        1st Qu.:-0.0300  
##  Median :  -2.325   Median :10.00        Median : 0.1300  
##  Mean   :   1.574   Mean   :13.61        Mean   : 0.1673  
##  3rd Qu.:  79.259   3rd Qu.:19.00        3rd Qu.: 0.3500  
##  Max.   : 154.952   Max.   :42.00        Max.   : 2.2200  
##  gyros_dumbbell_y   gyros_dumbbell_z  accel_dumbbell_x  accel_dumbbell_y 
##  Min.   :-1.99000   Min.   :-2.3000   Min.   :-237.00   Min.   :-182.00  
##  1st Qu.:-0.14000   1st Qu.:-0.3100   1st Qu.: -50.00   1st Qu.:  -9.00  
##  Median : 0.03000   Median :-0.1300   Median :  -8.00   Median :  40.00  
##  Mean   : 0.04093   Mean   :-0.1428   Mean   : -27.91   Mean   :  51.56  
##  3rd Qu.: 0.21000   3rd Qu.: 0.0300   3rd Qu.:  11.00   3rd Qu.: 109.00  
##  Max.   : 2.63000   Max.   : 1.7200   Max.   : 216.00   Max.   : 287.00  
##  accel_dumbbell_z  magnet_dumbbell_x magnet_dumbbell_y magnet_dumbbell_z
##  Min.   :-334.00   Min.   :-643.0    Min.   :-742.0    Min.   :-245.00  
##  1st Qu.:-141.00   1st Qu.:-537.0    1st Qu.: 232.0    1st Qu.: -46.00  
##  Median :  -1.00   Median :-478.0    Median : 308.0    Median :  12.00  
##  Mean   : -37.42   Mean   :-331.6    Mean   : 224.2    Mean   :  45.62  
##  3rd Qu.:  38.00   3rd Qu.:-307.0    3rd Qu.: 393.0    3rd Qu.:  93.00  
##  Max.   : 318.00   Max.   : 582.0    Max.   : 632.0    Max.   : 452.00  
##   roll_forearm     pitch_forearm     yaw_forearm      total_accel_forearm
##  Min.   :-180.00   Min.   :-72.50   Min.   :-180.00   Min.   : 0.00      
##  1st Qu.:  -1.86   1st Qu.:  0.00   1st Qu.: -71.20   1st Qu.:29.00      
##  Median :  20.60   Median :  8.95   Median :   0.00   Median :35.00      
##  Mean   :  33.04   Mean   : 10.59   Mean   :  18.03   Mean   :34.52      
##  3rd Qu.: 140.00   3rd Qu.: 28.30   3rd Qu.: 109.00   3rd Qu.:41.00      
##  Max.   : 180.00   Max.   : 88.70   Max.   : 180.00   Max.   :79.00      
##  gyros_forearm_x  gyros_forearm_y   gyros_forearm_z  accel_forearm_x  
##  Min.   :-2.870   Min.   :-6.4400   Min.   :-4.120   Min.   :-479.00  
##  1st Qu.:-0.210   1st Qu.:-1.4500   1st Qu.:-0.160   1st Qu.:-179.00  
##  Median : 0.030   Median : 0.0500   Median : 0.100   Median : -56.00  
##  Mean   : 0.156   Mean   : 0.1107   Mean   : 0.157   Mean   : -61.44  
##  3rd Qu.: 0.560   3rd Qu.: 1.6900   3rd Qu.: 0.510   3rd Qu.:  77.00  
##  Max.   : 3.520   Max.   : 6.1200   Max.   : 4.100   Max.   : 365.00  
##  accel_forearm_y  accel_forearm_z   magnet_forearm_x  magnet_forearm_y
##  Min.   :-632.0   Min.   :-446.00   Min.   :-1280.0   Min.   :-882    
##  1st Qu.:  46.0   1st Qu.:-182.00   1st Qu.: -609.0   1st Qu.: -24    
##  Median : 191.0   Median : -38.00   Median : -363.0   Median : 581    
##  Mean   : 158.2   Mean   : -54.92   Mean   : -303.6   Mean   : 368    
##  3rd Qu.: 310.0   3rd Qu.:  27.00   3rd Qu.:  -54.0   3rd Qu.: 733    
##  Max.   : 591.0   Max.   : 287.00   Max.   :  672.0   Max.   :1450    
##  magnet_forearm_z classe  
##  Min.   :-964.0   A:1674  
##  1st Qu.: 178.0   B:1139  
##  Median : 503.0   C:1026  
##  Mean   : 390.8   D: 964  
##  3rd Qu.: 650.0   E:1082  
##  Max.   :1080.0
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
##     2     4     6     8    13    18    22    29    30    31    33    43 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##    49    50    51    54    57    58    60    67    70    75    77    78 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##    80    81    83    85    88    90    96    97    99   100   103   105 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   109   113   117   119   127   132   133   139   147   153   155   160 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   163   167   170   173   174   176   177   202   211   212   215   217 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   225   227   230   231   234   236   242   245   248   250   251   253 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   255   256   261   264   265   267   269   271   274   275   280   281 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   282   288   292   296   311   317   320   326   329   337   339   340 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   343   346   352   355   357   360   362   368   370   374   375   376 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   377   379   380   384   385   386   387   391   395   399   403   405 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   407   413   417   419   425   428   430   440   441   442   444   447 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   448   451   456   462   463   464   466   467   472   476   477   482 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   484   485   490   491   502   504   506   508   509   513   514   520 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   530   532   533   539   546   549   550   551   552   554   560   561 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   562   564   571   579   593   594   602   603   607   608   610   611 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   612   613   616   620   621   623   625   627   630   632   636   647 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   648   654   658   659   661   663   664   666   671   673   675   679 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   680   681   685   687   693   694   697   698   701   705   707   708 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   712   714   715   718   720   723   724   726   729   734   736   738 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   742   747   748   752   753   763   764   765   768   774   776   781 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   795   798   801   804   809   819   820   826   829   833   836   840 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   841   847   848   850   852   855   860   861   863   869   871   874 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   877   878   880   883   885   887   901   902   907   908   912   918 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   920   922   927   931   932   933   934   935   937   940   945   949 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   955   958   960   962   967   976   980   984   996   999  1005  1009 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1010  1011  1017  1021  1023  1026  1028  1030  1033  1035  1038  1041 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1042  1046  1053  1054  1060  1065  1067  1070  1071  1072  1076  1078 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1082  1084  1087  1093  1097  1100  1101  1102  1111  1114  1120  1125 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1129  1130  1131  1133  1135  1136  1137  1139  1142  1143  1144  1146 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1148  1155  1156  1159  1161  1162  1165  1167  1168  1173  1175  1176 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1181  1182  1183  1186  1187  1192  1193  1194  1196  1200  1208  1210 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1213  1216  1217  1218  1225  1227  1229  1233  1234  1236  1238  1241 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1242  1253  1259  1270  1271  1275  1284  1287  1288  1290  1296  1297 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1302  1304  1310  1311  1321  1323  1325  1327  1336  1343  1345  1351 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1352  1355  1357  1358  1363  1364  1369  1370  1374  1381  1382  1383 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1384  1388  1389  1393  1395  1407  1408  1410  1427  1429  1432  1433 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1435  1438  1447  1451  1453  1467  1470  1484  1487  1492  1494  1495 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1504  1505  1506  1509  1512  1513  1514  1521  1529  1537  1541  1543 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1544  1545  1551  1553  1554  1555  1557  1561  1562  1564  1566  1567 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1576  1578  1579  1584  1585  1587  1591  1595  1596  1607  1608  1617 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1618  1628  1631  1632  1633  1634  1638  1639  1651  1662  1667  1674 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1676  1678  1680  1683  1684  1698  1699  1700  1702  1704  1707  1711 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1723  1727  1728  1730  1738  1739  1744  1745  1746  1750  1751  1754 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1755  1756  1761  1763  1764  1769  1771  1773  1782  1783  1785  1786 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1788  1789  1790  1791  1793  1801  1808  1809  1811  1824  1828  1830 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1834  1835  1838  1843  1844  1846  1847  1850  1851  1852  1857  1860 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1861  1863  1864  1866  1868  1873  1878  1880  1892  1897  1899  1903 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1904  1905  1906  1912  1917  1922  1923  1930  1933  1935  1937  1938 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1939  1947  1952  1960  1963  1967  1968  1970  1972  1975  1977  1978 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1981  1982  1986  1988  1989  1992  1995  1997  1998  1999  2002  2003 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2004  2005  2008  2009  2011  2014  2015  2016  2022  2026  2027  2029 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2034  2036  2039  2040  2045  2046  2047  2053  2058  2062  2063  2067 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2075  2077  2078  2080  2086  2089  2103  2111  2113  2116  2117  2119 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2124  2128  2129  2132  2133  2137  2138  2141  2142  2143  2146  2148 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2151  2152  2153  2159  2160  2162  2173  2174  2175  2179  2182  2191 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2194  2197  2204  2206  2209  2210  2213  2215  2219  2221  2224  2226 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2229  2230  2233  2237  2241  2246  2249  2259  2261  2264  2265  2268 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2269  2274  2284  2288  2289  2294  2296  2297  2298  2300  2301  2313 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2315  2316  2324  2327  2331  2333  2334  2336  2340  2343  2345  2346 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2349  2351  2352  2358  2365  2366  2368  2373  2379  2381  2386  2390 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2391  2393  2395  2398  2399  2407  2409  2411  2412  2416  2419  2421 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2422  2427  2434  2435  2436  2442  2446  2452  2457  2460  2462  2463 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2465  2467  2472  2474  2482  2485  2490  2491  2494  2500  2510  2512 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2516  2518  2519  2522  2525  2528  2532  2534  2536  2544  2546  2547 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2548  2549  2556  2557  2560  2561  2562  2565  2566  2567  2570  2573 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2576  2579  2580  2581  2582  2583  2586  2587  2588  2589  2590  2595 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2597  2598  2599  2602  2605  2609  2610  2612  2614  2622  2627  2628 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2629  2632  2635  2637  2638  2639  2643  2644  2651  2656  2657  2659 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2664  2669  2672  2674  2675  2678  2684  2689  2691  2692  2694  2697 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2700  2705  2711  2722  2724  2729  2737  2739  2747  2752  2753  2756 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2763  2765  2770  2784  2786  2787  2792  2798  2799  2800  2802  2803 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2805  2812  2816  2819  2823  2825  2826  2831  2833  2837  2838  2839 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2844  2852  2856  2860  2861  2863  2866  2867  2871  2872  2876  2878 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2882  2885  2889  2892  2894  2901  2906  2907  2915  2927  2928  2930 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2931  2933  2934  2935  2936  2939  2946  2949  2950  2956  2962  2967 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2977  2986  2992  2995  2997  3007  3018  3023  3024  3025  3033  3034 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3037  3041  3044  3048  3050  3051  3053  3055  3061  3063  3064  3070 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3071  3075  3076  3077  3079  3084  3087  3088  3093  3097  3098  3099 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3101  3104  3106  3110  3111  3114  3117  3118  3119  3127  3129  3130 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3132  3134  3135  3141  3144  3145  3148  3151  3154  3156  3158  3161 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3165  3167  3172  3182  3188  3190  3209  3210  3211  3215  3225  3227 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3229  3231  3234  3235  3237  3238  3243  3246  3251  3258  3259  3261 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3269  3274  3276  3279  3287  3288  3289  3292  3298  3301  3304  3314 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3315  3321  3323  3331  3333  3336  3337  3339  3341  3343  3346  3351 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3353  3356  3358  3360  3361  3369  3372  3373  3376  3394  3401  3402 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3406  3414  3415  3417  3421  3424  3427  3432  3433  3434  3437  3440 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3444  3445  3452  3456  3457  3460  3462  3463  3467  3468  3470  3471 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3473  3476  3478  3487  3488  3490  3493  3497  3499  3507  3509  3513 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3514  3515  3525  3527  3528  3529  3530  3533  3541  3542  3545  3548 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3552  3553  3554  3556  3558  3560  3563  3564  3572  3573  3574  3576 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3578  3582  3599  3604  3605  3607  3612  3613  3615  3616  3620  3621 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3622  3623  3628  3629  3630  3631  3632  3639  3640  3641  3646  3647 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3650  3651  3656  3658  3662  3664  3671  3673  3677  3679  3694  3696 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3697  3699  3702  3704  3705  3708  3710  3713  3719  3721  3726  3741 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3745  3749  3751  3755  3757  3763  3765  3766  3770  3771  3772  3774 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3776  3778  3788  3793  3795  3796  3804  3805  3806  3807  3808  3809 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3822  3828  3829  3830  3831  3837  3845  3846  3848  3851  3853  3858 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3862  3866  3868  3879  3886  3888  3889  3890  3893  3896  3898  3904 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3909  3910  3912  3919  3921  3924  3932  3933  3934  3936  3941  3946 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3951  3952  3953  3954  3955  3961  3968  3973  3976  3977  3980  3981 
##     A     B     B     A     A     A     A     A     A     A     A     A 
##  3985  3986  3988  3990  3991  3997  3999  4002  4006  4007  4008  4009 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4021  4022  4025  4026  4033  4035  4036  4043  4044  4054  4055  4057 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4058  4060  4064  4070  4073  4079  4080  4081  4086  4087  4091  4092 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4100  4105  4109  4112  4113  4114  4118  4119  4124  4126  4130  4131 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4139  4142  4145  4146  4149  4150  4151  4153  4154  4156  4158  4161 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4162  4173  4176  4177  4178  4179  4181  4182  4186  4200  4202  4204 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4209  4210  4211  4216  4217  4220  4221  4226  4227  4237  4249  4250 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4251  4253  4260  4271  4276  4278  4281  4282  4283  4286  4287  4288 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4289  4292  4294  4299  4300  4301  4304  4311  4313  4317  4318  4319 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4322  4323  4325  4330  4336  4342  4344  4348  4350  4353  4357  4359 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4362  4363  4366  4368  4372  4373  4375  4378  4379  4381  4384  4389 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4390  4391  4394  4400  4401  4405  4409  4411  4414  4415  4416  4417 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4424  4431  4434  4436  4438  4440  4442  4446  4453  4455  4459  4460 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4464  4466  4469  4470  4471  4472  4478  4487  4489  4495  4497  4498 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4499  4505  4518  4519  4535  4542  4548  4555  4560  4562  4568  4570 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4571  4576  4577  4581  4582  4583  4585  4588  4593  4594  4596  4598 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4602  4603  4606  4607  4611  4613  4614  4616  4623  4626  4632  4636 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4638  4640  4641  4643  4648  4649  4653  4663  4672  4673  4676  4681 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4692  4700  4707  4718  4720  4721  4722  4730  4732  4735  4736  4737 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4744  4745  4747  4751  4753  4754  4756  4763  4768  4771  4772  4773 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4775  4776  4777  4785  4787  4788  4790  4798  4799  4801  4804  4808 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4811  4814  4821  4833  4834  4835  4842  4847  4858  4860  4874  4876 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4880  4885  4889  4891  4898  4901  4905  4909  4911  4915  4924  4926 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4930  4943  4944  4946  4948  4950  4951  4965  4966  4969  4971  4978 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4982  4984  4987  4989  4992  5001  5002  5006  5008  5009  5010  5011 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5012  5013  5015  5016  5017  5018  5027  5035  5037  5042  5043  5046 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5053  5054  5062  5072  5077  5080  5086  5090  5094  5097  5099  5100 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5102  5104  5106  5108  5110  5112  5113  5115  5118  5119  5128  5132 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5135  5136  5143  5146  5148  5150  5153  5154  5157  5159  5161  5162 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5163  5165  5166  5168  5170  5175  5176  5177  5178  5180  5181  5186 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5188  5191  5200  5202  5203  5205  5207  5208  5210  5218  5223  5226 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5227  5236  5237  5239  5240  5242  5246  5247  5248  5251  5253  5257 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5260  5262  5274  5277  5283  5285  5292  5293  5296  5299  5303  5304 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5305  5311  5316  5320  5329  5330  5338  5341  5342  5344  5350  5351 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5366  5368  5370  5374  5381  5383  5386  5389  5393  5396  5400  5401 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5402  5404  5405  5408  5410  5413  5415  5419  5422  5423  5427  5429 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5431  5441  5446  5453  5454  5460  5465  5466  5467  5470  5471  5472 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5473  5481  5493  5497  5498  5499  5501  5502  5507  5509  5514  5518 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5522  5524  5528  5531  5532  5537  5541  5547  5550  5551  5553  5558 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5563  5568  5576  5577  5578  5580  5582  5583  5586  5592  5593  5604 
##     A     A     A     A     A     A     A     A     B     B     B     B 
##  5617  5620  5622  5625  5628  5630  5638  5640  5648  5650  5652  5653 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  5664  5666  5674  5684  5687  5691  5693  5694  5695  5699  5704  5705 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  5709  5714  5715  5718  5724  5725  5729  5734  5735  5736  5740  5741 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  5742  5743  5745  5747  5748  5752  5755  5757  5760  5762  5764  5765 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  5766  5769  5771  5772  5775  5778  5781  5786  5791  5797  5801  5803 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  5805  5806  5808  5813  5818  5824  5826  5827  5830  5834  5836  5838 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  5840  5841  5843  5847  5850  5856  5858  5861  5867  5869  5874  5875 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  5882  5886  5889  5891  5893  5894  5896  5897  5903  5904  5906  5910 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  5911  5918  5920  5921  5931  5933  5938  5942  5948  5949  5957  5965 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  5966  5968  5970  5971  5975  5977  5979  5985  5988  6012  6013  6016 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6020  6026  6028  6031  6032  6033  6035  6036  6043  6049  6052  6062 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6072  6074  6075  6076  6080  6082  6085  6088  6095  6097  6098  6102 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6103  6116  6122  6123  6129  6131  6133  6134  6135  6143  6144  6146 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6150  6151  6152  6154  6155  6161  6164  6165  6174  6175  6176  6178 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6179  6181  6182  6187  6188  6193  6194  6199  6200  6201  6202  6205 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6206  6210  6220  6222  6225  6228  6233  6235  6240  6245  6249  6250 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6252  6253  6259  6260  6262  6263  6267  6268  6273  6282  6287  6291 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6297  6298  6299  6301  6302  6307  6308  6320  6326  6332  6333  6335 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6340  6344  6346  6352  6354  6356  6359  6366  6368  6375  6376  6380 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6381  6382  6386  6387  6389  6390  6391  6394  6397  6398  6400  6414 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6417  6421  6423  6426  6428  6429  6436  6437  6446  6448  6452  6453 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6457  6458  6463  6466  6467  6469  6470  6472  6475  6480  6484  6489 
##     B     B     B     B     B     B     B     A     B     B     B     B 
##  6497  6501  6502  6506  6518  6521  6523  6529  6530  6540  6541  6544 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6545  6546  6548  6551  6552  6554  6556  6564  6565  6566  6569  6580 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6581  6582  6583  6586  6587  6589  6590  6591  6592  6593  6596  6597 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6604  6608  6610  6613  6615  6618  6623  6624  6625  6626  6632  6639 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6640  6644  6645  6652  6655  6656  6657  6668  6669  6680  6684  6686 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6689  6693  6695  6697  6699  6703  6705  6708  6719  6720  6723  6727 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6729  6730  6737  6738  6740  6742  6743  6744  6745  6750  6753  6754 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6758  6759  6760  6764  6767  6769  6770  6779  6783  6791  6796  6799 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6801  6803  6807  6810  6815  6816  6820  6821  6825  6826  6828  6830 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6836  6837  6843  6847  6849  6850  6856  6865  6867  6874  6885  6886 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6891  6894  6900  6901  6902  6905  6909  6914  6918  6920  6923  6929 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6930  6931  6932  6936  6937  6941  6945  6948  6955  6956  6957  6962 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6965  6966  6967  6970  6973  6974  6976  6980  6984  6985  6987  6988 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6997  7000  7001  7010  7011  7015  7016  7021  7028  7030  7033  7036 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7042  7043  7045  7049  7052  7054  7055  7058  7065  7073  7084  7089 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7090  7092  7095  7097  7099  7102  7103  7106  7107  7108  7109  7113 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7114  7118  7127  7131  7135  7142  7143  7146  7147  7148  7150  7154 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7155  7156  7158  7159  7165  7173  7177  7178  7181  7183  7189  7190 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7193  7199  7202  7205  7207  7209  7211  7219  7221  7225  7233  7235 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7243  7244  7245  7249  7255  7260  7262  7272  7274  7277  7282  7284 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7288  7289  7290  7297  7298  7300  7313  7321  7322  7326  7331  7332 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7334  7335  7342  7345  7350  7352  7353  7355  7363  7365  7368  7369 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7370  7372  7375  7379  7380  7382  7394  7400  7402  7403  7411  7414 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7415  7417  7418  7420  7424  7427  7428  7431  7435  7436  7442  7449 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7472  7473  7475  7477  7478  7479  7485  7487  7493  7505  7508  7511 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7512  7516  7517  7522  7527  7528  7529  7531  7535  7536  7550  7553 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7554  7570  7575  7583  7584  7585  7587  7591  7592  7596  7600  7606 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7607  7613  7616  7617  7620  7622  7629  7630  7633  7635  7640  7645 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7648  7655  7656  7657  7658  7661  7662  7667  7669  7685  7686  7691 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7693  7694  7698  7702  7710  7711  7712  7718  7723  7724  7731  7735 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7741  7742  7743  7744  7748  7752  7754  7760  7763  7765  7772  7783 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7789  7790  7796  7803  7805  7810  7812  7818  7819  7821  7823  7827 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7828  7832  7838  7839  7841  7842  7853  7855  7856  7859  7860  7862 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7863  7866  7868  7869  7878  7879  7881  7888  7892  7893  7895  7902 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7908  7911  7916  7918  7921  7928  7934  7935  7936  7939  7947  7948 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7956  7957  7959  7961  7963  7965  7967  7975  7976  7979  7981  7991 
##     B     A     B     A     B     B     B     B     B     B     B     B 
##  7993  7996  7999  8001  8003  8005  8012  8015  8021  8022  8032  8033 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8034  8037  8041  8046  8047  8051  8054  8057  8061  8064  8069  8073 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8074  8085  8088  8089  8091  8092  8093  8096  8098  8099  8102  8106 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8108  8111  8112  8113  8114  8121  8124  8125  8128  8133  8143  8148 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8154  8156  8157  8166  8167  8168  8172  8179  8180  8181  8183  8185 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8197  8201  8202  8207  8210  8211  8212  8215  8226  8229  8232  8233 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8234  8238  8240  8241  8242  8256  8259  8261  8263  8265  8266  8267 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8270  8272  8274  8279  8280  8284  8285  8291  8294  8298  8299  8301 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8304  8305  8308  8311  8317  8318  8329  8331  8332  8333  8335  8337 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8338  8340  8345  8348  8354  8356  8359  8364  8367  8371  8374  8375 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8376  8380  8384  8385  8387  8389  8394  8397  8400  8402  8407  8411 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8412  8413  8416  8417  8419  8424  8429  8433  8436  8438  8439  8440 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8444  8445  8447  8450  8451  8457  8460  8462  8464  8465  8471  8480 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8485  8489  8490  8493  8498  8499  8500  8502  8504  8505  8508  8512 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8514  8518  8523  8528  8529  8530  8535  8536  8544  8545  8551  8555 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8564  8567  8573  8574  8575  8577  8578  8580  8582  8584  8589  8591 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8592  8594  8596  8608  8611  8616  8617  8620  8624  8626  8628  8633 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8636  8640  8642  8646  8653  8654  8655  8664  8666  8673  8677  8684 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8685  8689  8691  8694  8695  8697  8698  8700  8701  8705  8707  8710 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8712  8718  8719  8720  8721  8724  8727  8728  8732  8736  8738  8745 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8749  8753  8755  8757  8758  8763  8765  8766  8769  8770  8773  8774 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8779  8780  8783  8784  8785  8787  8788  8793  8794  8800  8802  8807 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8810  8813  8814  8815  8824  8827  8828  8836  8837  8839  8840  8850 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8853  8854  8855  8860  8861  8863  8864  8870  8873  8875  8882  8883 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8885  8889  8891  8892  8894  8896  8905  8908  8913  8916  8917  8920 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8921  8922  8923  8930  8932  8939  8941  8943  8953  8958  8960  8961 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8964  8966  8968  8975  8979  8981  8985  8988  8991  8995  8997  9002 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  9006  9018  9019  9021  9023  9025  9026  9034  9044  9049  9053  9054 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  9055  9056  9057  9067  9071  9072  9075  9077  9080  9081  9086  9092 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  9095  9097  9104  9112  9116  9120  9122  9129  9135  9138  9141  9142 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  9144  9145  9147  9148  9149  9150  9152  9153  9154  9157  9159  9160 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  9161  9163  9168  9170  9175  9179  9182  9183  9191  9193  9195  9202 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  9216  9223  9226  9227  9229  9233  9239  9244  9247  9249  9251  9252 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  9253  9257  9263  9264  9267  9269  9283  9285  9286  9287  9289  9294 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  9295  9297  9302  9304  9305  9306  9309  9311  9312  9313  9317  9318 
##     B     B     B     B     B     B     B     B     C     B     B     B 
##  9324  9331  9341  9347  9352  9353  9355  9356  9359  9360  9361  9369 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  9370  9374  9375  9376  9377  9380  9384  9387  9393  9395  9397  9398 
##     B     B     B     B     B     C     C     C     C     C     C     C 
##  9399  9405  9406  9409  9410  9416  9420  9427  9431  9434  9436  9438 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9440  9442  9446  9448  9457  9458  9466  9469  9476  9481  9483  9491 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9494  9495  9498  9499  9501  9505  9506  9509  9514  9516  9519  9521 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9522  9528  9529  9530  9533  9538  9539  9551  9553  9557  9565  9566 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9569  9570  9578  9581  9592  9595  9597  9599  9601  9602  9604  9607 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9609  9611  9616  9619  9622  9628  9639  9645  9650  9654  9657  9661 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9663  9665  9668  9671  9686  9689  9690  9693  9698  9706  9707  9712 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9713  9718  9720  9721  9738  9739  9740  9749  9752  9753  9757  9758 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9761  9762  9772  9775  9779  9789  9792  9794  9798  9802  9803  9810 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9814  9818  9819  9820  9822  9830  9844  9845  9848  9851  9855  9863 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9865  9866  9867  9868  9873  9874  9875  9876  9879  9881  9886  9889 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9893  9895  9897  9898  9900  9901  9903  9907  9913  9920  9927  9933 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9952  9953  9956  9957  9961  9968  9972  9975  9989  9991  9998 10002 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10018 10020 10021 10030 10031 10032 10035 10036 10037 10041 10047 10049 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10051 10054 10060 10061 10069 10070 10071 10092 10095 10097 10099 10100 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10105 10107 10109 10110 10111 10115 10119 10123 10125 10126 10129 10130 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10131 10135 10137 10141 10144 10146 10148 10149 10156 10159 10160 10163 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10164 10167 10168 10173 10175 10176 10178 10179 10180 10181 10185 10186 
##     C     C     C     C     C     B     C     C     C     C     C     C 
## 10189 10190 10192 10195 10203 10204 10208 10210 10217 10225 10226 10228 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10237 10238 10240 10242 10243 10244 10246 10247 10249 10251 10253 10256 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10262 10266 10268 10281 10283 10284 10291 10296 10298 10299 10301 10302 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10310 10311 10312 10321 10323 10325 10329 10332 10336 10338 10342 10344 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10345 10349 10357 10359 10361 10363 10365 10366 10370 10371 10373 10374 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10375 10377 10378 10380 10381 10382 10386 10388 10393 10394 10397 10398 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10402 10403 10404 10411 10413 10416 10420 10423 10429 10430 10440 10447 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10449 10455 10456 10460 10461 10470 10482 10485 10488 10492 10493 10494 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10495 10496 10500 10503 10506 10509 10510 10517 10520 10523 10524 10525 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10534 10537 10539 10540 10541 10544 10546 10554 10555 10556 10561 10562 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10565 10566 10567 10568 10569 10570 10574 10577 10579 10581 10584 10587 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10589 10591 10593 10594 10597 10598 10599 10603 10604 10606 10607 10608 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10611 10615 10616 10623 10626 10627 10628 10637 10638 10642 10644 10648 
##     C     C     C     C     C     C     B     C     C     C     C     C 
## 10649 10650 10654 10655 10661 10662 10665 10666 10669 10670 10673 10674 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10676 10677 10678 10679 10684 10685 10688 10693 10697 10703 10716 10718 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10719 10730 10731 10733 10734 10741 10742 10747 10752 10755 10758 10759 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10762 10763 10766 10767 10771 10774 10779 10786 10787 10798 10800 10801 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10803 10807 10811 10814 10817 10821 10826 10829 10837 10839 10844 10848 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10855 10863 10867 10871 10874 10876 10878 10879 10882 10883 10885 10886 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10888 10894 10901 10903 10904 10906 10907 10911 10919 10921 10922 10926 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10929 10937 10954 10955 10956 10960 10962 10965 10968 10977 10978 10985 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10988 10989 10993 10994 10996 10997 11001 11004 11005 11007 11008 11022 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11023 11024 11027 11028 11029 11030 11031 11032 11035 11036 11037 11043 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11044 11045 11054 11057 11059 11064 11066 11070 11071 11077 11078 11079 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11086 11087 11090 11095 11101 11102 11111 11113 11116 11118 11119 11121 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11127 11129 11133 11144 11149 11150 11153 11154 11165 11168 11170 11177 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11179 11184 11189 11192 11200 11202 11203 11205 11208 11213 11215 11216 
##     C     C     C     C     C     C     C     C     B     C     C     C 
## 11221 11222 11227 11234 11236 11238 11240 11241 11247 11252 11256 11258 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11262 11268 11270 11277 11280 11281 11288 11289 11290 11293 11300 11311 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11312 11315 11316 11324 11325 11327 11329 11330 11332 11333 11338 11353 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11356 11357 11360 11361 11367 11368 11373 11374 11379 11380 11385 11387 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11390 11395 11396 11398 11401 11403 11411 11412 11421 11427 11435 11445 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11447 11449 11450 11454 11455 11459 11461 11463 11464 11466 11471 11475 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11476 11477 11478 11480 11481 11484 11487 11488 11491 11495 11500 11502 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11511 11513 11518 11520 11521 11525 11527 11530 11531 11540 11541 11544 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11549 11552 11561 11562 11568 11569 11570 11574 11575 11581 11582 11592 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11593 11599 11601 11603 11604 11613 11621 11622 11625 11627 11628 11632 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11636 11640 11645 11648 11655 11659 11661 11664 11670 11673 11674 11677 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11679 11680 11682 11683 11684 11685 11687 11690 11691 11698 11699 11701 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11703 11705 11707 11711 11713 11715 11718 11719 11723 11727 11732 11733 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11737 11740 11742 11743 11744 11745 11747 11749 11756 11765 11767 11768 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11769 11770 11771 11772 11777 11781 11782 11783 11784 11786 11792 11795 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11804 11814 11824 11825 11826 11827 11829 11838 11840 11845 11848 11850 
##     C     C     D     D     D     D     C     C     C     C     C     C 
## 11851 11857 11858 11860 11862 11863 11869 11880 11882 11890 11891 11902 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11910 11911 11915 11916 11917 11919 11920 11923 11924 11926 11928 11935 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11939 11940 11945 11946 11948 11949 11952 11954 11955 11956 11959 11960 
##     C     C     C     C     C     C     C     C     C     C     B     B 
## 11963 11965 11968 11978 11982 11983 11984 11986 11991 11996 11997 12002 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12004 12005 12007 12009 12010 12012 12013 12017 12018 12020 12022 12024 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12025 12026 12027 12030 12033 12040 12042 12043 12045 12046 12047 12054 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12055 12062 12065 12066 12067 12068 12071 12078 12080 12083 12084 12089 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12090 12101 12103 12104 12108 12109 12110 12111 12112 12113 12119 12120 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12126 12128 12129 12131 12135 12138 12139 12152 12153 12155 12161 12163 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12165 12170 12172 12173 12174 12178 12184 12191 12193 12195 12197 12200 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12205 12206 12208 12210 12213 12218 12225 12229 12231 12236 12237 12238 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12241 12246 12250 12252 12255 12257 12260 12261 12264 12265 12266 12267 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12269 12272 12278 12289 12295 12304 12306 12308 12318 12323 12326 12329 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12331 12340 12341 12344 12346 12347 12349 12350 12351 12352 12353 12367 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12375 12376 12381 12389 12391 12397 12404 12405 12406 12407 12408 12409 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12410 12411 12416 12418 12419 12422 12427 12431 12433 12434 12439 12444 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12448 12450 12465 12467 12470 12475 12477 12479 12480 12483 12486 12487 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12491 12492 12493 12494 12496 12502 12505 12508 12516 12519 12524 12527 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12530 12534 12537 12538 12540 12542 12546 12551 12556 12572 12574 12579 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12580 12584 12588 12590 12594 12595 12601 12602 12609 12614 12619 12620 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12621 12622 12623 12627 12629 12631 12635 12636 12638 12639 12656 12658 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12662 12665 12672 12681 12683 12687 12689 12692 12696 12698 12699 12700 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12702 12703 12705 12708 12710 12711 12712 12717 12721 12722 12724 12742 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12753 12755 12757 12763 12766 12775 12776 12778 12780 12789 12792 12801 
##     C     C     C     C     C     C     C     C     C     C     C     D 
## 12803 12806 12817 12821 12824 12827 12828 12830 12834 12835 12836 12837 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 12843 12846 12847 12854 12856 12862 12868 12872 12873 12874 12881 12886 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 12887 12892 12900 12909 12914 12915 12916 12922 12926 12939 12940 12942 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 12944 12945 12954 12957 12958 12960 12961 12967 12968 12969 12970 12971 
##     D     D     D     D     D     C     C     C     C     C     C     D 
## 12981 12984 12986 12987 12991 12996 12999 13000 13003 13004 13005 13007 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13010 13016 13020 13023 13024 13030 13035 13038 13046 13051 13053 13054 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13055 13062 13069 13070 13071 13073 13078 13081 13086 13088 13091 13100 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13102 13105 13110 13116 13119 13126 13131 13132 13139 13143 13149 13150 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13153 13155 13160 13161 13166 13170 13172 13177 13180 13184 13185 13188 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13192 13193 13195 13199 13200 13202 13204 13206 13208 13211 13212 13215 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13220 13224 13225 13226 13228 13232 13233 13240 13241 13245 13247 13248 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13249 13250 13253 13254 13257 13264 13266 13270 13272 13274 13278 13286 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13293 13297 13299 13324 13325 13326 13329 13330 13334 13340 13344 13345 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13347 13348 13356 13361 13362 13366 13368 13374 13381 13387 13389 13390 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13398 13399 13404 13409 13411 13418 13422 13426 13428 13433 13435 13438 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13445 13449 13450 13454 13458 13459 13461 13464 13466 13467 13471 13474 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13484 13487 13488 13491 13499 13500 13501 13504 13506 13507 13513 13514 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13517 13518 13526 13528 13532 13534 13536 13540 13542 13547 13550 13553 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13556 13557 13560 13561 13563 13566 13567 13570 13574 13579 13591 13595 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13596 13598 13601 13602 13604 13609 13610 13612 13614 13615 13628 13634 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13641 13642 13643 13644 13647 13649 13651 13659 13661 13667 13671 13678 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13679 13680 13690 13691 13694 13696 13698 13706 13707 13711 13714 13719 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13723 13726 13727 13728 13729 13731 13734 13740 13749 13752 13755 13767 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13774 13777 13778 13779 13780 13781 13787 13789 13792 13794 13796 13797 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13798 13812 13813 13815 13818 13826 13830 13832 13844 13847 13849 13856 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13857 13859 13863 13864 13866 13871 13876 13881 13886 13888 13896 13897 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13898 13899 13903 13911 13914 13917 13934 13938 13939 13940 13942 13943 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13945 13949 13957 13963 13965 13967 13976 13981 13993 13994 13999 14001 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14003 14004 14011 14012 14014 14018 14019 14021 14023 14024 14028 14030 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14032 14037 14038 14039 14045 14049 14050 14054 14055 14059 14064 14067 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14069 14071 14077 14078 14081 14084 14086 14088 14090 14092 14093 14094 
##     D     D     D     D     D     C     D     D     C     C     C     C 
## 14098 14099 14105 14107 14108 14113 14125 14127 14130 14131 14132 14135 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14136 14137 14140 14142 14143 14145 14147 14148 14152 14154 14155 14159 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14160 14161 14165 14172 14176 14177 14178 14183 14185 14187 14191 14192 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14202 14203 14209 14210 14212 14213 14219 14220 14221 14222 14228 14229 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14234 14237 14239 14243 14244 14257 14258 14260 14261 14263 14266 14268 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14269 14270 14274 14278 14285 14289 14296 14298 14299 14300 14304 14306 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14312 14314 14316 14317 14325 14326 14327 14329 14332 14335 14339 14340 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14341 14344 14345 14348 14349 14350 14359 14363 14364 14366 14367 14368 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14372 14373 14376 14378 14379 14384 14385 14386 14387 14389 14391 14398 
##     C     C     D     D     D     D     D     D     D     C     D     D 
## 14409 14412 14415 14418 14419 14425 14432 14433 14434 14441 14442 14452 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14454 14455 14461 14462 14470 14473 14475 14477 14480 14481 14485 14487 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14489 14499 14502 14514 14518 14519 14520 14521 14523 14531 14533 14540 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14541 14545 14547 14550 14553 14560 14568 14570 14575 14579 14581 14582 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14584 14585 14592 14599 14601 14602 14611 14613 14618 14620 14628 14629 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14632 14633 14636 14643 14645 14647 14649 14657 14669 14677 14678 14681 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14682 14684 14686 14692 14695 14699 14713 14716 14722 14723 14727 14730 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14731 14733 14734 14740 14742 14745 14746 14747 14748 14749 14754 14755 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14758 14762 14763 14769 14774 14777 14779 14782 14787 14788 14789 14808 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14809 14810 14812 14814 14822 14824 14827 14831 14836 14837 14839 14840 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14841 14851 14855 14859 14864 14865 14868 14869 14871 14873 14874 14877 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14885 14889 14892 14893 14894 14896 14900 14906 14911 14912 14914 14918 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14919 14920 14927 14946 14951 14952 14953 14957 14959 14962 14963 14965 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14978 14981 14983 14988 14993 14999 15005 15016 15019 15023 15026 15027 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15030 15032 15036 15037 15041 15047 15048 15058 15064 15066 15068 15070 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15072 15078 15083 15086 15090 15094 15098 15100 15103 15104 15106 15107 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15109 15110 15116 15119 15121 15124 15125 15132 15134 15137 15146 15154 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15155 15158 15159 15160 15162 15163 15166 15167 15168 15170 15172 15173 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15175 15177 15179 15181 15182 15183 15185 15190 15199 15200 15207 15208 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15209 15210 15217 15218 15229 15231 15233 15234 15239 15246 15256 15257 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15263 15271 15273 15274 15275 15278 15280 15284 15294 15295 15300 15302 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15304 15306 15308 15309 15310 15315 15321 15324 15325 15326 15327 15328 
##     D     D     D     D     D     D     C     C     C     C     C     D 
## 15334 15335 15336 15337 15338 15340 15341 15344 15348 15349 15352 15357 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15358 15359 15361 15362 15363 15364 15368 15369 15370 15371 15372 15374 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15375 15376 15379 15380 15381 15383 15386 15393 15395 15396 15400 15403 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15407 15412 15414 15415 15416 15420 15426 15427 15433 15435 15437 15438 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15440 15444 15447 15450 15452 15458 15460 15464 15466 15468 15470 15479 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15481 15482 15483 15491 15498 15500 15503 15506 15513 15514 15519 15525 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15530 15532 15538 15542 15546 15549 15553 15559 15563 15564 15565 15568 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15569 15572 15574 15579 15583 15586 15593 15601 15602 15605 15606 15614 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15617 15619 15621 15622 15624 15629 15631 15632 15633 15636 15642 15643 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15646 15647 15652 15655 15657 15664 15669 15670 15673 15675 15676 15681 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15684 15689 15691 15693 15697 15707 15722 15723 15733 15735 15749 15751 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15752 15753 15754 15755 15756 15759 15761 15769 15772 15776 15784 15792 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15794 15797 15799 15803 15804 15807 15809 15810 15813 15814 15817 15818 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15821 15824 15827 15838 15841 15842 15845 15847 15848 15850 15852 15854 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15855 15858 15862 15866 15874 15875 15888 15889 15892 15895 15897 15898 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15899 15901 15905 15906 15910 15912 15913 15916 15917 15919 15921 15928 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15944 15947 15951 15953 15954 15956 15957 15959 15960 15963 15964 15967 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15968 15969 15971 15972 15973 15974 15981 15984 15988 15995 15998 16000 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 16006 16008 16011 16017 16026 16029 16035 16037 16038 16039 16042 16044 
##     D     D     D     E     E     E     E     E     E     E     E     E 
## 16047 16049 16054 16057 16058 16059 16060 16061 16063 16064 16069 16074 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16077 16080 16082 16097 16098 16100 16103 16104 16105 16111 16113 16114 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16116 16120 16121 16124 16130 16132 16138 16139 16141 16143 16147 16148 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16151 16154 16157 16167 16172 16174 16176 16177 16180 16184 16186 16189 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16190 16191 16192 16194 16195 16201 16202 16214 16219 16220 16221 16222 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16224 16227 16228 16232 16234 16240 16251 16252 16253 16258 16262 16265 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16270 16274 16277 16281 16288 16294 16297 16300 16302 16309 16319 16321 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16322 16329 16330 16335 16337 16346 16352 16353 16355 16356 16360 16370 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16377 16384 16388 16393 16395 16396 16400 16409 16411 16414 16415 16416 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16418 16421 16422 16430 16435 16436 16437 16438 16439 16441 16442 16443 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16444 16445 16447 16454 16455 16460 16461 16463 16467 16469 16479 16481 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16483 16484 16485 16486 16488 16490 16493 16498 16499 16506 16509 16513 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16514 16519 16521 16523 16524 16529 16534 16538 16546 16550 16557 16559 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16561 16563 16570 16573 16575 16581 16584 16585 16586 16587 16588 16597 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16599 16600 16604 16609 16610 16619 16624 16625 16626 16627 16628 16632 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16633 16636 16637 16638 16640 16641 16644 16656 16661 16665 16667 16676 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16681 16682 16684 16692 16693 16699 16700 16703 16704 16708 16712 16714 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16717 16723 16724 16731 16733 16739 16741 16742 16744 16745 16746 16750 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16754 16755 16773 16777 16778 16795 16805 16808 16809 16811 16813 16816 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16818 16825 16828 16833 16837 16840 16843 16845 16849 16850 16851 16852 
##     E     E     E     E     E     E     E     E     D     E     D     E 
## 16855 16872 16873 16874 16877 16881 16885 16893 16894 16895 16896 16898 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16901 16904 16909 16912 16914 16917 16918 16919 16922 16924 16925 16930 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16937 16942 16943 16945 16953 16954 16956 16958 16960 16962 16966 16973 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16982 16985 16992 16993 17001 17003 17007 17022 17023 17028 17029 17030 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17031 17032 17033 17039 17041 17042 17043 17050 17054 17057 17058 17060 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17063 17065 17066 17067 17068 17071 17075 17083 17084 17086 17091 17093 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17097 17101 17102 17104 17106 17109 17121 17123 17128 17132 17134 17140 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17141 17143 17158 17163 17165 17166 17172 17175 17177 17178 17183 17187 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17193 17196 17197 17203 17212 17213 17215 17218 17222 17225 17227 17229 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17230 17232 17233 17234 17236 17238 17246 17253 17256 17258 17259 17263 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17266 17268 17269 17276 17278 17280 17281 17288 17293 17297 17298 17300 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17305 17308 17311 17317 17323 17324 17329 17332 17333 17335 17336 17341 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17344 17345 17355 17356 17358 17361 17364 17372 17373 17377 17384 17386 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17387 17388 17389 17392 17393 17394 17402 17403 17404 17408 17416 17419 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17424 17431 17432 17434 17435 17437 17440 17443 17445 17449 17450 17452 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17453 17455 17465 17469 17472 17474 17477 17481 17483 17492 17499 17503 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17505 17507 17508 17511 17513 17516 17519 17521 17522 17523 17526 17528 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17529 17531 17532 17536 17539 17542 17546 17553 17557 17558 17560 17562 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17563 17568 17569 17574 17575 17579 17582 17588 17590 17594 17595 17596 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17602 17604 17605 17608 17610 17616 17617 17619 17620 17621 17622 17624 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17625 17630 17632 17634 17638 17639 17640 17641 17643 17650 17654 17655 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17657 17663 17668 17669 17670 17671 17677 17686 17688 17689 17690 17693 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17700 17710 17720 17721 17722 17723 17724 17733 17734 17741 17743 17747 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17749 17750 17755 17763 17766 17767 17775 17781 17783 17786 17791 17793 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17796 17797 17800 17802 17803 17804 17807 17808 17811 17813 17818 17819 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17820 17821 17825 17826 17829 17832 17835 17837 17838 17843 17844 17845 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17846 17847 17851 17854 17857 17863 17865 17868 17869 17870 17872 17884 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17885 17888 17890 17896 17898 17901 17906 17907 17908 17913 17920 17921 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17929 17930 17931 17935 17950 17952 17966 17967 17969 17973 17975 17980 
##     E     E     E     E     E     E     E     D     E     E     E     E 
## 17984 17985 17986 17987 17993 17995 17999 18000 18001 18004 18006 18008 
##     E     E     E     E     E     D     E     E     E     E     E     E 
## 18012 18014 18022 18024 18025 18029 18036 18037 18053 18056 18061 18066 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18068 18074 18077 18084 18085 18087 18099 18101 18103 18106 18107 18108 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18112 18114 18124 18133 18135 18140 18143 18146 18154 18155 18162 18163 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18165 18170 18172 18181 18182 18183 18188 18190 18197 18198 18199 18203 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18204 18207 18210 18213 18214 18218 18219 18224 18226 18227 18229 18232 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18234 18235 18238 18240 18242 18246 18247 18248 18253 18256 18260 18262 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18267 18271 18275 18277 18278 18279 18280 18290 18291 18293 18295 18296 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18316 18319 18323 18329 18330 18333 18339 18340 18343 18345 18352 18356 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18360 18361 18362 18363 18364 18368 18369 18377 18379 18381 18382 18386 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18388 18391 18393 18394 18397 18398 18400 18402 18407 18411 18413 18417 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18419 18440 18443 18446 18448 18450 18452 18463 18472 18474 18476 18477 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18478 18481 18487 18488 18495 18499 18510 18512 18514 18515 18516 18517 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18521 18522 18537 18550 18551 18556 18563 18571 18572 18583 18590 18597 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18604 18606 18610 18612 18614 18615 18617 18618 18628 18639 18644 18649 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18651 18652 18653 18656 18658 18659 18664 18665 18672 18677 18679 18685 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18687 18690 18691 18694 18697 18700 18701 18702 18704 18709 18710 18714 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18716 18717 18721 18723 18726 18728 18735 18738 18740 18742 18745 18748 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18755 18756 18761 18765 18769 18771 18773 18774 18778 18779 18781 18785 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18787 18788 18789 18792 18794 18800 18802 18806 18807 18811 18813 18814 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18815 18816 18819 18824 18825 18828 18831 18835 18836 18842 18845 18851 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18853 18858 18863 18864 18866 18869 18870 18871 18874 18876 18879 18880 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18883 18885 18886 18887 18889 18893 18900 18903 18904 18906 18907 18915 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18920 18927 18935 18937 18939 18942 18945 18949 18950 18952 18955 18957 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18959 18966 18971 18981 18986 18987 18991 18992 18993 18994 18998 18999 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19004 19007 19009 19010 19014 19015 19016 19018 19020 19021 19025 19026 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19028 19031 19033 19035 19038 19039 19042 19047 19049 19051 19055 19057 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19058 19059 19061 19066 19069 19073 19077 19078 19081 19090 19094 19096 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19100 19102 19103 19106 19107 19110 19111 19114 19117 19120 19123 19126 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19128 19132 19133 19135 19141 19149 19151 19153 19166 19173 19174 19175 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19176 19178 19192 19194 19210 19213 19214 19215 19217 19224 19228 19230 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19232 19236 19237 19240 19243 19245 19246 19250 19253 19255 19256 19260 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19262 19264 19265 19274 19281 19286 19296 19298 19301 19305 19306 19307 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19309 19311 19312 19315 19323 19332 19335 19336 19339 19340 19341 19343 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19345 19347 19348 19352 19355 19357 19362 19363 19364 19366 19368 19375 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19381 19384 19388 19393 19396 19398 19406 19409 19415 19421 19422 19425 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19426 19429 19430 19437 19439 19441 19446 19447 19452 19454 19455 19456 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19461 19462 19466 19468 19472 19483 19484 19489 19496 19498 19502 19503 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19507 19509 19510 19513 19514 19515 19518 19522 19525 19537 19541 19544 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19560 19576 19584 19588 19591 19593 19594 19599 19601 19602 19605 19610 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19615 19617 19618 19620 19622 
##     E     E     E     E     E 
## Levels: A B C D E
```
