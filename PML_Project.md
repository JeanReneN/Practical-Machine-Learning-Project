####Practical Machine Learning - Human Activity Recognition Prediction Project#####
######By Jean Rene Ndeki######
######Wednesday, November 19, 2014######

####I. Introduction####
The Human Activity Recognition (HAR) project consists of predicting participants' activities using exercise data. The potential applications of the project include elderly monitoring, life log systems for monitoring energy expenditure, weight-loss program support, and digital assistants for weight lifting exercises. Data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants are gathered and used for modeling. The question the project tries to answer is how to predict the manner in which the participants will perform their exercise. 

First, a working directory is locally set to store the project data and files.

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

####II. Data####
The data source is "http://groupware.les.inf.puc-rio.br/har".
The files are downloaded from the listed url to the local directory.

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

The study design requires a cross validation set. The clean training set is partitioned into a training set (70%) and cross validation set (30%). Three files (training, cross validation, and test) will be used for the study.


```r
inTrain <- createDataPartition(y = training_cm$classe, p = 0.7, list = FALSE)

training   <- training_cm[inTrain, ]        # training set

validation <- training_cm[-inTrain, ]       # cross validaton set

testing    <- testing_cm                    # testing set
```

####III. Features ####
The features are shown in appendix VIII.1 (Exploratory Data Analysis). Notice, the training and cross validation sets have similar features, 53 variables. The test set differs from the training and cross validation sets in two fields. The outcome "classe" exists in the training and cross validation sets. The  problem id is only in the test set.The study suggests that any of the other variables may be used to predict with.

####IV. Algorithm and Prediction####
The random forest algorithm is applied.The model is created using the training set. Multiple deep decision trees, trained on different parts of the same training set, are averaged with the goal of reducing the variance. Bootstrap aggregating or bagging is applied: sampling with replacement, decision tree training, and predictions' average for the majority vote selection. The prediction model is defined as follows:


```r
modFit <- randomForest(classe ~ ., data = training)
```

####V. Cross Validation and Testing####
The cross validation is performed using the model developed from the training set and applying it to the validation set.


```r
#Cross validation
predcv <- predict(modFit, validation)
```
The Cross validation results figure in appendix VIII.2.

The confusion matrix is produced below.

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
##          B    6 1131    2    0    0
##          C    0    8 1016    2    0
##          D    0    0   22  941    1
##          E    0    0    0    4 1078
## 
## Overall Statistics
##                                        
##                Accuracy : 0.992        
##                  95% CI : (0.99, 0.994)
##     No Information Rate : 0.285        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.99         
##  Mcnemar's Test P-Value : NA           
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.996    0.993    0.977    0.994    0.999
## Specificity             1.000    0.998    0.998    0.995    0.999
## Pos Pred Value          1.000    0.993    0.990    0.976    0.996
## Neg Pred Value          0.999    0.998    0.995    0.999    1.000
## Prevalence              0.285    0.194    0.177    0.161    0.183
## Detection Rate          0.284    0.192    0.173    0.160    0.183
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       0.998    0.996    0.987    0.995    0.999
```

The model is evaluated using test data. 

```r
predt <- predict(modFit, testing)
```
The test results figure below.

```r
print(predt)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

####VI. Results and Analysis####
1. The model accuracy is 0.9995 with 95% confidence
2. The P Value is small (< 2.2e-16), indicating a statistically significant test
3. The Kappa statistic is high (0.9994), suggesting a near complete agreement (close to 1)
4. The sensitivity and specificity are high (>.99%) for each class (A,B,C,D).

Details of the analysis for each class sensitivity and specificity are shown above.

####VII. Conclusions####
The Human Activity Recognition (HAR) predictive model is highly accuracy, as evidenced by the cross validation, and generalized in the 20 test cases. The manner in which the participants will do their exercise can be predicted with 99.95% accuracy and 95% confidence. The HAR random forest predictive model is performant.

####VIII. Appendix####

####VIII.1 Exploratory Data Analysis####


```r
#Training set
str(training)
```

```
## 'data.frame':	13737 obs. of  53 variables:
##  $ roll_belt           : num  1.41 1.41 1.42 1.43 1.45 1.43 1.42 1.42 1.48 1.51 ...
##  $ pitch_belt          : num  8.07 8.07 8.09 8.16 8.18 8.18 8.2 8.21 8.15 8.12 ...
##  $ yaw_belt            : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt    : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ gyros_belt_x        : num  0 0.02 0.02 0.02 0.03 0.02 0.02 0.02 0 0 ...
##  $ gyros_belt_y        : num  0 0 0 0 0 0 0 0 0 0 ...
##  $ gyros_belt_z        : num  -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 0 -0.02 0 -0.02 ...
##  $ accel_belt_x        : int  -21 -22 -22 -20 -21 -22 -22 -22 -21 -21 ...
##  $ accel_belt_y        : int  4 4 3 2 2 2 4 4 4 4 ...
##  $ accel_belt_z        : int  22 22 21 24 23 23 21 21 23 22 ...
##  $ magnet_belt_x       : int  -3 -7 -4 1 -5 -2 -3 -8 0 -6 ...
##  $ magnet_belt_y       : int  599 608 599 602 596 602 606 598 592 598 ...
##  $ magnet_belt_z       : int  -313 -311 -311 -312 -317 -319 -309 -310 -305 -317 ...
##  $ roll_arm            : num  -128 -128 -128 -128 -128 -128 -128 -128 -129 -129 ...
##  $ pitch_arm           : num  22.5 22.5 21.9 21.7 21.5 21.5 21.4 21.4 21.3 21.3 ...
##  $ yaw_arm             : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
##  $ total_accel_arm     : int  34 34 34 34 34 34 34 34 34 34 ...
##  $ gyros_arm_x         : num  0 0.02 0 0.02 0.02 0.02 0.02 0.02 0.02 0.02 ...
##  $ gyros_arm_y         : num  0 -0.02 -0.03 -0.03 -0.03 -0.03 -0.02 0 0 0 ...
##  $ gyros_arm_z         : num  -0.02 -0.02 0 -0.02 0 0 -0.02 -0.03 -0.03 -0.02 ...
##  $ accel_arm_x         : int  -288 -290 -289 -288 -290 -288 -287 -288 -289 -289 ...
##  $ accel_arm_y         : int  109 110 111 109 110 111 111 111 109 110 ...
##  $ accel_arm_z         : int  -123 -125 -125 -122 -123 -123 -124 -124 -121 -122 ...
##  $ magnet_arm_x        : int  -368 -369 -373 -369 -366 -363 -372 -371 -367 -371 ...
##  $ magnet_arm_y        : int  337 337 336 341 339 343 338 331 340 337 ...
##  $ magnet_arm_z        : int  516 513 509 518 509 520 509 523 509 512 ...
##  $ roll_dumbbell       : num  13.1 13.1 13.1 13.2 13.1 ...
##  $ pitch_dumbbell      : num  -70.5 -70.6 -70.2 -70.4 -70.6 ...
##  $ yaw_dumbbell        : num  -84.9 -84.7 -85.1 -84.9 -84.7 ...
##  $ total_accel_dumbbell: int  37 37 37 37 37 37 37 37 37 37 ...
##  $ gyros_dumbbell_x    : num  0 0 0 0 0 0 0 0.02 0 0 ...
##  $ gyros_dumbbell_y    : num  -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 ...
##  $ gyros_dumbbell_z    : num  0 0 0 0 0 0 -0.02 -0.02 0 0 ...
##  $ accel_dumbbell_x    : int  -234 -233 -232 -232 -233 -233 -234 -234 -233 -233 ...
##  $ accel_dumbbell_y    : int  47 47 47 47 47 47 48 48 48 47 ...
##  $ accel_dumbbell_z    : int  -271 -269 -270 -269 -269 -270 -269 -268 -271 -272 ...
##  $ magnet_dumbbell_x   : int  -559 -555 -551 -549 -564 -554 -552 -554 -554 -551 ...
##  $ magnet_dumbbell_y   : int  293 296 295 292 299 291 302 295 297 296 ...
##  $ magnet_dumbbell_z   : num  -65 -64 -70 -65 -64 -65 -69 -68 -73 -56 ...
##  $ roll_forearm        : num  28.4 28.3 27.9 27.7 27.6 27.5 27.2 27.2 27.1 27.1 ...
##  $ pitch_forearm       : num  -63.9 -63.9 -63.9 -63.8 -63.8 -63.8 -63.9 -63.9 -64 -64 ...
##  $ yaw_forearm         : num  -153 -153 -152 -152 -152 -152 -151 -151 -151 -151 ...
##  $ total_accel_forearm : int  36 36 36 36 36 36 36 36 36 36 ...
##  $ gyros_forearm_x     : num  0.03 0.02 0.02 0.03 0.02 0.02 0 0 0.02 0.02 ...
##  $ gyros_forearm_y     : num  0 0 0 0 -0.02 0.02 0 -0.02 0 -0.02 ...
##  $ gyros_forearm_z     : num  -0.02 -0.02 -0.02 -0.02 -0.02 -0.03 -0.03 -0.03 0 0 ...
##  $ accel_forearm_x     : int  192 192 195 193 193 191 193 193 194 192 ...
##  $ accel_forearm_y     : int  203 203 205 204 205 203 205 202 204 204 ...
##  $ accel_forearm_z     : int  -215 -216 -215 -214 -214 -215 -215 -214 -215 -213 ...
##  $ magnet_forearm_x    : int  -17 -18 -18 -16 -17 -11 -15 -14 -13 -13 ...
##  $ magnet_forearm_y    : num  654 661 659 653 657 657 655 659 656 653 ...
##  $ magnet_forearm_z    : num  476 473 470 476 465 478 472 478 471 481 ...
##  $ classe              : Factor w/ 5 levels "A","B","C","D",..: 1 1 1 1 1 1 1 1 1 1 ...
```

```r
head(training)
```

```
##    roll_belt pitch_belt yaw_belt total_accel_belt gyros_belt_x
## 1       1.41       8.07    -94.4                3         0.00
## 2       1.41       8.07    -94.4                3         0.02
## 7       1.42       8.09    -94.4                3         0.02
## 9       1.43       8.16    -94.4                3         0.02
## 11      1.45       8.18    -94.4                3         0.03
## 12      1.43       8.18    -94.4                3         0.02
##    gyros_belt_y gyros_belt_z accel_belt_x accel_belt_y accel_belt_z
## 1             0        -0.02          -21            4           22
## 2             0        -0.02          -22            4           22
## 7             0        -0.02          -22            3           21
## 9             0        -0.02          -20            2           24
## 11            0        -0.02          -21            2           23
## 12            0        -0.02          -22            2           23
##    magnet_belt_x magnet_belt_y magnet_belt_z roll_arm pitch_arm yaw_arm
## 1             -3           599          -313     -128      22.5    -161
## 2             -7           608          -311     -128      22.5    -161
## 7             -4           599          -311     -128      21.9    -161
## 9              1           602          -312     -128      21.7    -161
## 11            -5           596          -317     -128      21.5    -161
## 12            -2           602          -319     -128      21.5    -161
##    total_accel_arm gyros_arm_x gyros_arm_y gyros_arm_z accel_arm_x
## 1               34        0.00        0.00       -0.02        -288
## 2               34        0.02       -0.02       -0.02        -290
## 7               34        0.00       -0.03        0.00        -289
## 9               34        0.02       -0.03       -0.02        -288
## 11              34        0.02       -0.03        0.00        -290
## 12              34        0.02       -0.03        0.00        -288
##    accel_arm_y accel_arm_z magnet_arm_x magnet_arm_y magnet_arm_z
## 1          109        -123         -368          337          516
## 2          110        -125         -369          337          513
## 7          111        -125         -373          336          509
## 9          109        -122         -369          341          518
## 11         110        -123         -366          339          509
## 12         111        -123         -363          343          520
##    roll_dumbbell pitch_dumbbell yaw_dumbbell total_accel_dumbbell
## 1          13.05         -70.49       -84.87                   37
## 2          13.13         -70.64       -84.71                   37
## 7          13.13         -70.25       -85.10                   37
## 9          13.15         -70.43       -84.92                   37
## 11         13.13         -70.64       -84.71                   37
## 12         13.10         -70.46       -84.89                   37
##    gyros_dumbbell_x gyros_dumbbell_y gyros_dumbbell_z accel_dumbbell_x
## 1                 0            -0.02                0             -234
## 2                 0            -0.02                0             -233
## 7                 0            -0.02                0             -232
## 9                 0            -0.02                0             -232
## 11                0            -0.02                0             -233
## 12                0            -0.02                0             -233
##    accel_dumbbell_y accel_dumbbell_z magnet_dumbbell_x magnet_dumbbell_y
## 1                47             -271              -559               293
## 2                47             -269              -555               296
## 7                47             -270              -551               295
## 9                47             -269              -549               292
## 11               47             -269              -564               299
## 12               47             -270              -554               291
##    magnet_dumbbell_z roll_forearm pitch_forearm yaw_forearm
## 1                -65         28.4         -63.9        -153
## 2                -64         28.3         -63.9        -153
## 7                -70         27.9         -63.9        -152
## 9                -65         27.7         -63.8        -152
## 11               -64         27.6         -63.8        -152
## 12               -65         27.5         -63.8        -152
##    total_accel_forearm gyros_forearm_x gyros_forearm_y gyros_forearm_z
## 1                   36            0.03            0.00           -0.02
## 2                   36            0.02            0.00           -0.02
## 7                   36            0.02            0.00           -0.02
## 9                   36            0.03            0.00           -0.02
## 11                  36            0.02           -0.02           -0.02
## 12                  36            0.02            0.02           -0.03
##    accel_forearm_x accel_forearm_y accel_forearm_z magnet_forearm_x
## 1              192             203            -215              -17
## 2              192             203            -216              -18
## 7              195             205            -215              -18
## 9              193             204            -214              -16
## 11             193             205            -214              -17
## 12             191             203            -215              -11
##    magnet_forearm_y magnet_forearm_z classe
## 1               654              476      A
## 2               661              473      A
## 7               659              470      A
## 9               653              476      A
## 11              657              465      A
## 12              657              478      A
```

```r
summary(training)
```

```
##    roll_belt        pitch_belt        yaw_belt       total_accel_belt
##  Min.   :-28.80   Min.   :-55.80   Min.   :-180.00   Min.   : 0.0    
##  1st Qu.:  1.09   1st Qu.:  1.79   1st Qu.: -88.30   1st Qu.: 3.0    
##  Median :113.00   Median :  5.28   Median : -14.20   Median :17.0    
##  Mean   : 64.08   Mean   :  0.38   Mean   : -11.89   Mean   :11.3    
##  3rd Qu.:123.00   3rd Qu.: 14.90   3rd Qu.:   8.89   3rd Qu.:18.0    
##  Max.   :162.00   Max.   : 60.30   Max.   : 179.00   Max.   :29.0    
##   gyros_belt_x      gyros_belt_y      gyros_belt_z    accel_belt_x    
##  Min.   :-1.0400   Min.   :-0.6400   Min.   :-1.46   Min.   :-120.00  
##  1st Qu.:-0.0300   1st Qu.: 0.0000   1st Qu.:-0.20   1st Qu.: -21.00  
##  Median : 0.0300   Median : 0.0200   Median :-0.10   Median : -15.00  
##  Mean   :-0.0058   Mean   : 0.0394   Mean   :-0.13   Mean   :  -5.73  
##  3rd Qu.: 0.1100   3rd Qu.: 0.1100   3rd Qu.:-0.02   3rd Qu.:  -5.00  
##  Max.   : 2.2200   Max.   : 0.6400   Max.   : 1.62   Max.   :  85.00  
##   accel_belt_y  accel_belt_z  magnet_belt_x   magnet_belt_y magnet_belt_z 
##  Min.   :-69   Min.   :-275   Min.   :-52.0   Min.   :354   Min.   :-623  
##  1st Qu.:  3   1st Qu.:-162   1st Qu.:  9.0   1st Qu.:581   1st Qu.:-375  
##  Median : 33   Median :-151   Median : 35.0   Median :601   Median :-320  
##  Mean   : 30   Mean   : -72   Mean   : 55.5   Mean   :594   Mean   :-346  
##  3rd Qu.: 61   3rd Qu.:  28   3rd Qu.: 59.0   3rd Qu.:610   3rd Qu.:-306  
##  Max.   :164   Max.   : 105   Max.   :485.0   Max.   :669   Max.   : 289  
##     roll_arm        pitch_arm         yaw_arm        total_accel_arm
##  Min.   :-180.0   Min.   :-88.80   Min.   :-180.00   Min.   : 1.0   
##  1st Qu.: -31.3   1st Qu.:-25.50   1st Qu.: -42.80   1st Qu.:17.0   
##  Median :   0.0   Median :  0.00   Median :   0.00   Median :27.0   
##  Mean   :  18.2   Mean   : -4.46   Mean   :  -0.21   Mean   :25.5   
##  3rd Qu.:  77.2   3rd Qu.: 11.30   3rd Qu.:  46.30   3rd Qu.:33.0   
##  Max.   : 180.0   Max.   : 88.50   Max.   : 180.00   Max.   :66.0   
##   gyros_arm_x     gyros_arm_y      gyros_arm_z      accel_arm_x    
##  Min.   :-6.37   Min.   :-3.440   Min.   :-2.330   Min.   :-383.0  
##  1st Qu.:-1.33   1st Qu.:-0.800   1st Qu.:-0.070   1st Qu.:-241.0  
##  Median : 0.06   Median :-0.240   Median : 0.230   Median : -45.0  
##  Mean   : 0.04   Mean   :-0.254   Mean   : 0.268   Mean   : -60.4  
##  3rd Qu.: 1.56   3rd Qu.: 0.140   3rd Qu.: 0.720   3rd Qu.:  83.0  
##  Max.   : 4.87   Max.   : 2.810   Max.   : 3.020   Max.   : 435.0  
##   accel_arm_y      accel_arm_z      magnet_arm_x   magnet_arm_y 
##  Min.   :-302.0   Min.   :-636.0   Min.   :-584   Min.   :-392  
##  1st Qu.: -54.0   1st Qu.:-143.0   1st Qu.:-302   1st Qu.: -10  
##  Median :  14.0   Median : -47.0   Median : 290   Median : 201  
##  Mean   :  33.1   Mean   : -71.7   Mean   : 191   Mean   : 156  
##  3rd Qu.: 141.0   3rd Qu.:  23.0   3rd Qu.: 636   3rd Qu.: 323  
##  Max.   : 308.0   Max.   : 292.0   Max.   : 782   Max.   : 583  
##   magnet_arm_z  roll_dumbbell    pitch_dumbbell    yaw_dumbbell    
##  Min.   :-597   Min.   :-153.7   Min.   :-149.6   Min.   :-147.11  
##  1st Qu.: 134   1st Qu.: -17.2   1st Qu.: -40.9   1st Qu.: -77.60  
##  Median : 443   Median :  48.5   Median : -20.9   Median :  -4.07  
##  Mean   : 306   Mean   :  24.4   Mean   : -10.6   Mean   :   1.52  
##  3rd Qu.: 545   3rd Qu.:  68.1   3rd Qu.:  17.9   3rd Qu.:  79.59  
##  Max.   : 694   Max.   : 153.4   Max.   : 137.0   Max.   : 154.75  
##  total_accel_dumbbell gyros_dumbbell_x  gyros_dumbbell_y gyros_dumbbell_z
##  Min.   : 0.0         Min.   :-204.00   Min.   :-2.10    Min.   : -2.4   
##  1st Qu.: 4.0         1st Qu.:  -0.03   1st Qu.:-0.14    1st Qu.: -0.3   
##  Median :10.0         Median :   0.13   Median : 0.03    Median : -0.1   
##  Mean   :13.7         Mean   :   0.16   Mean   : 0.05    Mean   : -0.1   
##  3rd Qu.:19.0         3rd Qu.:   0.35   3rd Qu.: 0.21    3rd Qu.:  0.0   
##  Max.   :58.0         Max.   :   2.22   Max.   :52.00    Max.   :317.0   
##  accel_dumbbell_x accel_dumbbell_y accel_dumbbell_z magnet_dumbbell_x
##  Min.   :-419.0   Min.   :-189.0   Min.   :-284.0   Min.   :-643     
##  1st Qu.: -51.0   1st Qu.:  -8.0   1st Qu.:-142.0   1st Qu.:-535     
##  Median :  -8.0   Median :  42.0   Median :  -1.0   Median :-478     
##  Mean   : -28.6   Mean   :  53.1   Mean   : -37.9   Mean   :-328     
##  3rd Qu.:  11.0   3rd Qu.: 111.0   3rd Qu.:  38.0   3rd Qu.:-299     
##  Max.   : 224.0   Max.   : 315.0   Max.   : 318.0   Max.   : 592     
##  magnet_dumbbell_y magnet_dumbbell_z  roll_forearm    pitch_forearm   
##  Min.   :-744      Min.   :-262.0    Min.   :-180.0   Min.   :-72.50  
##  1st Qu.: 231      1st Qu.: -45.0    1st Qu.:  -0.7   1st Qu.:  0.00  
##  Median : 311      Median :  14.0    Median :  22.4   Median :  9.31  
##  Mean   : 222      Mean   :  46.9    Mean   :  34.2   Mean   : 10.63  
##  3rd Qu.: 391      3rd Qu.:  96.0    3rd Qu.: 140.0   3rd Qu.: 28.20  
##  Max.   : 633      Max.   : 451.0    Max.   : 180.0   Max.   : 89.80  
##   yaw_forearm     total_accel_forearm gyros_forearm_x   gyros_forearm_y 
##  Min.   :-180.0   Min.   :  0.0       Min.   :-22.000   Min.   : -6.65  
##  1st Qu.: -67.9   1st Qu.: 29.0       1st Qu.: -0.220   1st Qu.: -1.46  
##  Median :   0.0   Median : 36.0       Median :  0.050   Median :  0.03  
##  Mean   :  19.6   Mean   : 34.7       Mean   :  0.152   Mean   :  0.09  
##  3rd Qu.: 110.0   3rd Qu.: 41.0       3rd Qu.:  0.550   3rd Qu.:  1.64  
##  Max.   : 180.0   Max.   :108.0       Max.   :  3.970   Max.   :311.00  
##  gyros_forearm_z  accel_forearm_x  accel_forearm_y accel_forearm_z 
##  Min.   : -8.09   Min.   :-496.0   Min.   :-585    Min.   :-410.0  
##  1st Qu.: -0.18   1st Qu.:-177.0   1st Qu.:  57    1st Qu.:-182.0  
##  Median :  0.08   Median : -56.0   Median : 202    Median : -38.0  
##  Mean   :  0.16   Mean   : -60.6   Mean   : 164    Mean   : -54.3  
##  3rd Qu.:  0.49   3rd Qu.:  77.0   3rd Qu.: 311    3rd Qu.:  27.0  
##  Max.   :231.00   Max.   : 389.0   Max.   : 923    Max.   : 287.0  
##  magnet_forearm_x magnet_forearm_y magnet_forearm_z classe  
##  Min.   :-1280    Min.   :-896     Min.   :-973     A:3906  
##  1st Qu.: -615    1st Qu.:   6     1st Qu.: 183     B:2658  
##  Median : -376    Median : 593     Median : 512     C:2396  
##  Mean   : -311    Mean   : 383     Mean   : 392     D:2252  
##  3rd Qu.:  -73    3rd Qu.: 738     3rd Qu.: 653     E:2525  
##  Max.   :  672    Max.   :1460     Max.   :1090
```

```r
#Cross Validation set
str(validation)
```

```
## 'data.frame':	5885 obs. of  53 variables:
##  $ roll_belt           : num  1.42 1.48 1.48 1.45 1.42 1.45 1.45 1.55 1.56 1.53 ...
##  $ pitch_belt          : num  8.07 8.05 8.07 8.06 8.13 8.17 8.2 8.08 8.1 8.11 ...
##  $ yaw_belt            : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.3 -94.4 ...
##  $ total_accel_belt    : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ gyros_belt_x        : num  0 0.02 0.02 0.02 0.02 0.03 0 0 0.02 0.03 ...
##  $ gyros_belt_y        : num  0 0 0.02 0 0 0 0 0.02 0 0 ...
##  $ gyros_belt_z        : num  -0.02 -0.03 -0.02 -0.02 -0.02 0 0 0 -0.02 0 ...
##  $ accel_belt_x        : int  -20 -22 -21 -21 -22 -21 -21 -21 -21 -19 ...
##  $ accel_belt_y        : int  5 3 2 4 4 4 2 5 4 4 ...
##  $ accel_belt_z        : int  23 21 24 21 21 22 22 21 21 21 ...
##  $ magnet_belt_x       : int  -2 -6 -6 0 -2 -3 -1 1 -4 -8 ...
##  $ magnet_belt_y       : int  600 604 600 603 603 609 597 600 606 605 ...
##  $ magnet_belt_z       : int  -305 -310 -302 -312 -313 -308 -310 -316 -311 -319 ...
##  $ roll_arm            : num  -128 -128 -128 -128 -128 -128 -129 -129 -129 -129 ...
##  $ pitch_arm           : num  22.5 22.1 22.1 22 21.8 21.6 21.4 21.2 20.7 20.7 ...
##  $ yaw_arm             : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
##  $ total_accel_arm     : int  34 34 34 34 34 34 34 34 34 34 ...
##  $ gyros_arm_x         : num  0.02 0.02 0 0.02 0.02 0.02 0.02 0.02 0.02 -0.02 ...
##  $ gyros_arm_y         : num  -0.02 -0.03 -0.03 -0.03 -0.02 -0.03 0 -0.02 -0.02 -0.02 ...
##  $ gyros_arm_z         : num  -0.02 0.02 0 0 0 -0.02 -0.03 -0.03 -0.02 0 ...
##  $ accel_arm_x         : int  -289 -289 -289 -289 -289 -288 -289 -288 -290 -289 ...
##  $ accel_arm_y         : int  110 111 111 111 111 110 111 108 110 109 ...
##  $ accel_arm_z         : int  -126 -123 -123 -122 -124 -124 -124 -124 -123 -123 ...
##  $ magnet_arm_x        : int  -368 -372 -374 -369 -372 -376 -374 -373 -373 -370 ...
##  $ magnet_arm_y        : int  344 344 337 342 338 334 342 336 333 340 ...
##  $ magnet_arm_z        : int  513 512 506 513 510 516 510 510 509 512 ...
##  $ roll_dumbbell       : num  12.9 13.4 13.4 13.4 12.8 ...
##  $ pitch_dumbbell      : num  -70.3 -70.4 -70.4 -70.8 -70.3 ...
##  $ yaw_dumbbell        : num  -85.1 -84.9 -84.9 -84.5 -85.1 ...
##  $ total_accel_dumbbell: int  37 37 37 37 37 37 37 36 37 37 ...
##  $ gyros_dumbbell_x    : num  0 0 0 0 0 0 0 0.02 0 0 ...
##  $ gyros_dumbbell_y    : num  -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 ...
##  $ gyros_dumbbell_z    : num  0 -0.02 0 0 0 0 0 -0.02 0 0 ...
##  $ accel_dumbbell_x    : int  -232 -232 -233 -234 -234 -235 -234 -231 -234 -234 ...
##  $ accel_dumbbell_y    : int  46 48 48 48 46 48 47 47 48 47 ...
##  $ accel_dumbbell_z    : int  -270 -269 -270 -269 -272 -270 -270 -268 -270 -271 ...
##  $ magnet_dumbbell_x   : int  -561 -552 -554 -558 -555 -558 -554 -557 -557 -555 ...
##  $ magnet_dumbbell_y   : int  298 303 292 294 300 291 294 292 294 290 ...
##  $ magnet_dumbbell_z   : num  -63 -60 -68 -66 -74 -69 -63 -62 -69 -68 ...
##  $ roll_forearm        : num  28.3 28.1 28 27.9 27.8 27.7 27.2 27 26.9 27.1 ...
##  $ pitch_forearm       : num  -63.9 -63.9 -63.9 -63.9 -63.8 -63.8 -63.9 -64 -63.8 -63.7 ...
##  $ yaw_forearm         : num  -152 -152 -152 -152 -152 -152 -151 -151 -151 -151 ...
##  $ total_accel_forearm : int  36 36 36 36 36 36 36 36 36 36 ...
##  $ gyros_forearm_x     : num  0.03 0.02 0.02 0.02 0.02 0.02 0 0.02 0.02 0.05 ...
##  $ gyros_forearm_y     : num  -0.02 -0.02 0 -0.02 -0.02 0 -0.02 0 -0.02 -0.03 ...
##  $ gyros_forearm_z     : num  0 0 -0.02 -0.03 0 -0.02 -0.02 -0.02 -0.02 0 ...
##  $ accel_forearm_x     : int  196 189 189 193 193 190 192 192 194 191 ...
##  $ accel_forearm_y     : int  204 206 206 203 205 205 201 206 206 202 ...
##  $ accel_forearm_z     : int  -213 -214 -214 -215 -213 -215 -214 -216 -214 -214 ...
##  $ magnet_forearm_x    : int  -18 -16 -17 -9 -9 -22 -16 -16 -10 -14 ...
##  $ magnet_forearm_y    : num  658 658 655 660 660 656 656 653 653 667 ...
##  $ magnet_forearm_z    : num  469 469 473 478 474 473 472 472 467 470 ...
##  $ classe              : Factor w/ 5 levels "A","B","C","D",..: 1 1 1 1 1 1 1 1 1 1 ...
```

```r
head(validation)
```

```
##    roll_belt pitch_belt yaw_belt total_accel_belt gyros_belt_x
## 3       1.42       8.07    -94.4                3         0.00
## 4       1.48       8.05    -94.4                3         0.02
## 5       1.48       8.07    -94.4                3         0.02
## 6       1.45       8.06    -94.4                3         0.02
## 8       1.42       8.13    -94.4                3         0.02
## 10      1.45       8.17    -94.4                3         0.03
##    gyros_belt_y gyros_belt_z accel_belt_x accel_belt_y accel_belt_z
## 3          0.00        -0.02          -20            5           23
## 4          0.00        -0.03          -22            3           21
## 5          0.02        -0.02          -21            2           24
## 6          0.00        -0.02          -21            4           21
## 8          0.00        -0.02          -22            4           21
## 10         0.00         0.00          -21            4           22
##    magnet_belt_x magnet_belt_y magnet_belt_z roll_arm pitch_arm yaw_arm
## 3             -2           600          -305     -128      22.5    -161
## 4             -6           604          -310     -128      22.1    -161
## 5             -6           600          -302     -128      22.1    -161
## 6              0           603          -312     -128      22.0    -161
## 8             -2           603          -313     -128      21.8    -161
## 10            -3           609          -308     -128      21.6    -161
##    total_accel_arm gyros_arm_x gyros_arm_y gyros_arm_z accel_arm_x
## 3               34        0.02       -0.02       -0.02        -289
## 4               34        0.02       -0.03        0.02        -289
## 5               34        0.00       -0.03        0.00        -289
## 6               34        0.02       -0.03        0.00        -289
## 8               34        0.02       -0.02        0.00        -289
## 10              34        0.02       -0.03       -0.02        -288
##    accel_arm_y accel_arm_z magnet_arm_x magnet_arm_y magnet_arm_z
## 3          110        -126         -368          344          513
## 4          111        -123         -372          344          512
## 5          111        -123         -374          337          506
## 6          111        -122         -369          342          513
## 8          111        -124         -372          338          510
## 10         110        -124         -376          334          516
##    roll_dumbbell pitch_dumbbell yaw_dumbbell total_accel_dumbbell
## 3          12.85         -70.28       -85.14                   37
## 4          13.43         -70.39       -84.87                   37
## 5          13.38         -70.43       -84.85                   37
## 6          13.38         -70.82       -84.47                   37
## 8          12.75         -70.35       -85.10                   37
## 10         13.33         -70.85       -84.45                   37
##    gyros_dumbbell_x gyros_dumbbell_y gyros_dumbbell_z accel_dumbbell_x
## 3                 0            -0.02             0.00             -232
## 4                 0            -0.02            -0.02             -232
## 5                 0            -0.02             0.00             -233
## 6                 0            -0.02             0.00             -234
## 8                 0            -0.02             0.00             -234
## 10                0            -0.02             0.00             -235
##    accel_dumbbell_y accel_dumbbell_z magnet_dumbbell_x magnet_dumbbell_y
## 3                46             -270              -561               298
## 4                48             -269              -552               303
## 5                48             -270              -554               292
## 6                48             -269              -558               294
## 8                46             -272              -555               300
## 10               48             -270              -558               291
##    magnet_dumbbell_z roll_forearm pitch_forearm yaw_forearm
## 3                -63         28.3         -63.9        -152
## 4                -60         28.1         -63.9        -152
## 5                -68         28.0         -63.9        -152
## 6                -66         27.9         -63.9        -152
## 8                -74         27.8         -63.8        -152
## 10               -69         27.7         -63.8        -152
##    total_accel_forearm gyros_forearm_x gyros_forearm_y gyros_forearm_z
## 3                   36            0.03           -0.02            0.00
## 4                   36            0.02           -0.02            0.00
## 5                   36            0.02            0.00           -0.02
## 6                   36            0.02           -0.02           -0.03
## 8                   36            0.02           -0.02            0.00
## 10                  36            0.02            0.00           -0.02
##    accel_forearm_x accel_forearm_y accel_forearm_z magnet_forearm_x
## 3              196             204            -213              -18
## 4              189             206            -214              -16
## 5              189             206            -214              -17
## 6              193             203            -215               -9
## 8              193             205            -213               -9
## 10             190             205            -215              -22
##    magnet_forearm_y magnet_forearm_z classe
## 3               658              469      A
## 4               658              469      A
## 5               655              473      A
## 6               660              478      A
## 8               660              474      A
## 10              656              473      A
```

```r
summary(validation)
```

```
##    roll_belt        pitch_belt        yaw_belt      total_accel_belt
##  Min.   :-28.90   Min.   :-54.40   Min.   :-179.0   Min.   : 0.0    
##  1st Qu.:  1.11   1st Qu.:  1.67   1st Qu.: -88.3   1st Qu.: 3.0    
##  Median :115.00   Median :  5.29   Median : -11.6   Median :17.0    
##  Mean   : 65.18   Mean   :  0.12   Mean   :  -9.6   Mean   :11.4    
##  3rd Qu.:123.00   3rd Qu.: 15.20   3rd Qu.:  15.3   3rd Qu.:18.0    
##  Max.   :162.00   Max.   : 60.10   Max.   : 179.0   Max.   :28.0    
##   gyros_belt_x      gyros_belt_y    gyros_belt_z     accel_belt_x   
##  Min.   :-0.8800   Min.   :-0.53   Min.   :-1.350   Min.   :-83.00  
##  1st Qu.:-0.0300   1st Qu.: 0.00   1st Qu.:-0.200   1st Qu.:-21.00  
##  Median : 0.0300   Median : 0.02   Median :-0.110   Median :-15.00  
##  Mean   :-0.0051   Mean   : 0.04   Mean   :-0.132   Mean   : -5.27  
##  3rd Qu.: 0.1100   3rd Qu.: 0.11   3rd Qu.: 0.000   3rd Qu.: -5.00  
##  Max.   : 2.2000   Max.   : 0.63   Max.   : 1.360   Max.   : 83.00  
##   accel_belt_y    accel_belt_z    magnet_belt_x   magnet_belt_y
##  Min.   :-38.0   Min.   :-269.0   Min.   :-48.0   Min.   :365  
##  1st Qu.:  3.0   1st Qu.:-162.0   1st Qu.:  8.0   1st Qu.:581  
##  Median : 38.0   Median :-154.0   Median : 34.0   Median :601  
##  Mean   : 30.5   Mean   : -73.9   Mean   : 55.8   Mean   :594  
##  3rd Qu.: 61.0   3rd Qu.:  27.0   3rd Qu.: 61.0   3rd Qu.:610  
##  Max.   : 90.0   Max.   : 103.0   Max.   :449.0   Max.   :673  
##  magnet_belt_z     roll_arm        pitch_arm         yaw_arm       
##  Min.   :-621   Min.   :-180.0   Min.   :-87.90   Min.   :-180.00  
##  1st Qu.:-374   1st Qu.: -32.9   1st Qu.:-26.70   1st Qu.: -43.40  
##  Median :-319   Median :   0.0   Median :  0.00   Median :   0.00  
##  Mean   :-345   Mean   :  17.0   Mean   : -4.96   Mean   :  -1.56  
##  3rd Qu.:-306   3rd Qu.:  77.6   3rd Qu.: 11.10   3rd Qu.:  44.60  
##  Max.   : 293   Max.   : 179.0   Max.   : 88.20   Max.   : 180.00  
##  total_accel_arm  gyros_arm_x      gyros_arm_y      gyros_arm_z    
##  Min.   : 1.0    Min.   :-6.340   Min.   :-3.260   Min.   :-2.280  
##  1st Qu.:17.0    1st Qu.:-1.300   1st Qu.:-0.800   1st Qu.:-0.070  
##  Median :27.0    Median : 0.100   Median :-0.260   Median : 0.250  
##  Mean   :25.4    Mean   : 0.049   Mean   :-0.265   Mean   : 0.273  
##  3rd Qu.:33.0    3rd Qu.: 1.590   3rd Qu.: 0.140   3rd Qu.: 0.720  
##  Max.   :64.0    Max.   : 4.820   Max.   : 2.840   Max.   : 2.950  
##   accel_arm_x      accel_arm_y      accel_arm_z      magnet_arm_x 
##  Min.   :-404.0   Min.   :-318.0   Min.   :-613.0   Min.   :-576  
##  1st Qu.:-243.0   1st Qu.: -54.0   1st Qu.:-144.0   1st Qu.:-296  
##  Median : -41.0   Median :  13.0   Median : -47.0   Median : 285  
##  Mean   : -59.9   Mean   :  31.3   Mean   : -70.3   Mean   : 193  
##  3rd Qu.:  85.0   3rd Qu.: 136.0   3rd Qu.:  23.0   3rd Qu.: 640  
##  Max.   : 437.0   Max.   : 296.0   Max.   : 227.0   Max.   : 780  
##   magnet_arm_y   magnet_arm_z  roll_dumbbell    pitch_dumbbell  
##  Min.   :-386   Min.   :-584   Min.   :-153.5   Min.   :-137.3  
##  1st Qu.:  -8   1st Qu.: 128   1st Qu.: -21.4   1st Qu.: -40.8  
##  Median : 202   Median : 446   Median :  47.4   Median : -21.0  
##  Mean   : 157   Mean   : 307   Mean   :  22.5   Mean   : -11.1  
##  3rd Qu.: 323   3rd Qu.: 545   3rd Qu.:  66.8   3rd Qu.:  16.8  
##  Max.   : 578   Max.   : 685   Max.   : 153.6   Max.   : 149.4  
##   yaw_dumbbell     total_accel_dumbbell gyros_dumbbell_x gyros_dumbbell_y 
##  Min.   :-150.87   Min.   : 0.0         Min.   :-1.860   Min.   :-1.9900  
##  1st Qu.: -77.73   1st Qu.: 5.0         1st Qu.:-0.030   1st Qu.:-0.1600  
##  Median :  -2.19   Median :10.0         Median : 0.130   Median : 0.0300  
##  Mean   :   2.04   Mean   :13.7         Mean   : 0.175   Mean   : 0.0402  
##  3rd Qu.:  79.96   3rd Qu.:19.0         3rd Qu.: 0.370   3rd Qu.: 0.2100  
##  Max.   : 154.95   Max.   :42.0         Max.   : 2.200   Max.   : 2.7300  
##  gyros_dumbbell_z accel_dumbbell_x accel_dumbbell_y accel_dumbbell_z
##  Min.   :-1.900   Min.   :-237.0   Min.   :-181.0   Min.   :-334.0  
##  1st Qu.:-0.310   1st Qu.: -50.0   1st Qu.:  -9.0   1st Qu.:-141.0  
##  Median :-0.130   Median :  -9.0   Median :  41.0   Median :  -1.0  
##  Mean   :-0.146   Mean   : -28.6   Mean   :  51.6   Mean   : -39.3  
##  3rd Qu.: 0.030   3rd Qu.:  10.0   3rd Qu.: 110.0   3rd Qu.:  38.0  
##  Max.   : 1.720   Max.   : 235.0   Max.   : 310.0   Max.   : 318.0  
##  magnet_dumbbell_x magnet_dumbbell_y magnet_dumbbell_z  roll_forearm    
##  Min.   :-638      Min.   :-3600     Min.   :-250.0    Min.   :-180.00  
##  1st Qu.:-537      1st Qu.:  232     1st Qu.: -45.0    1st Qu.:  -0.79  
##  Median :-482      Median :  309     Median :  12.0    Median :  19.40  
##  Mean   :-330      Mean   :  218     Mean   :  44.1    Mean   :  32.84  
##  3rd Qu.:-312      3rd Qu.:  388     3rd Qu.:  93.0    3rd Qu.: 140.00  
##  Max.   : 583      Max.   :  632     Max.   : 452.0    Max.   : 180.00  
##  pitch_forearm     yaw_forearm     total_accel_forearm gyros_forearm_x 
##  Min.   :-71.60   Min.   :-180.0   Min.   : 0.0        Min.   :-4.950  
##  1st Qu.:  0.00   1st Qu.: -70.7   1st Qu.:29.0        1st Qu.:-0.210  
##  Median :  8.99   Median :   0.0   Median :36.0        Median : 0.050  
##  Mean   : 10.89   Mean   :  18.3   Mean   :34.8        Mean   : 0.173  
##  3rd Qu.: 28.70   3rd Qu.: 110.0   3rd Qu.:41.0        3rd Qu.: 0.610  
##  Max.   : 87.40   Max.   : 180.0   Max.   :79.0        Max.   : 3.070  
##  gyros_forearm_y  gyros_forearm_z  accel_forearm_x  accel_forearm_y
##  Min.   :-7.020   Min.   :-7.940   Min.   :-498.0   Min.   :-632   
##  1st Qu.:-1.460   1st Qu.:-0.180   1st Qu.:-181.0   1st Qu.:  54   
##  Median : 0.030   Median : 0.070   Median : -59.0   Median : 199   
##  Mean   : 0.049   Mean   : 0.136   Mean   : -64.2   Mean   : 162   
##  3rd Qu.: 1.610   3rd Qu.: 0.490   3rd Qu.:  74.0   3rd Qu.: 315   
##  Max.   : 6.130   Max.   : 4.310   Max.   : 477.0   Max.   : 591   
##  accel_forearm_z  magnet_forearm_x magnet_forearm_y magnet_forearm_z
##  Min.   :-446.0   Min.   :-1280    Min.   :-892     Min.   :-966    
##  1st Qu.:-182.0   1st Qu.: -618    1st Qu.:  -5     1st Qu.: 209    
##  Median : -43.0   Median : -385    Median : 589     Median : 509    
##  Mean   : -57.7   Mean   : -316    Mean   : 374     Mean   : 397    
##  3rd Qu.:  23.0   3rd Qu.:  -77    3rd Qu.: 734     3rd Qu.: 651    
##  Max.   : 291.0   Max.   :  663    Max.   :1480     Max.   :1050    
##  classe  
##  A:1674  
##  B:1139  
##  C:1026  
##  D: 964  
##  E:1082  
## 
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
## 1         -326          385          481        -17.74          24.96
## 2         -325          447          434         54.48         -53.70
## 3         -264          474          413         57.07         -51.37
## 4         -173          257          633         43.11         -30.05
## 5         -170          275          617       -101.38         -53.44
## 6          396          176          516         62.19         -50.56
##   yaw_dumbbell total_accel_dumbbell gyros_dumbbell_x gyros_dumbbell_y
## 1       126.24                    9             0.64             0.06
## 2       -75.51                   31             0.34             0.05
## 3       -75.20                   29             0.39             0.14
## 4      -103.32                   18             0.10            -0.02
## 5       -14.20                    4             0.29            -0.47
## 6       -71.12                   29            -0.59             0.80
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
##    roll_belt        pitch_belt        yaw_belt     total_accel_belt
##  Min.   : -5.92   Min.   :-41.60   Min.   :-93.7   Min.   : 2.00   
##  1st Qu.:  0.91   1st Qu.:  3.01   1st Qu.:-88.6   1st Qu.: 3.00   
##  Median :  1.11   Median :  4.66   Median :-87.8   Median : 4.00   
##  Mean   : 31.31   Mean   :  5.82   Mean   :-59.3   Mean   : 7.55   
##  3rd Qu.: 32.51   3rd Qu.:  6.13   3rd Qu.:-63.5   3rd Qu.: 8.00   
##  Max.   :129.00   Max.   : 27.80   Max.   :162.0   Max.   :21.00   
##   gyros_belt_x     gyros_belt_y     gyros_belt_z     accel_belt_x   
##  Min.   :-0.500   Min.   :-0.050   Min.   :-0.480   Min.   :-48.00  
##  1st Qu.:-0.070   1st Qu.:-0.005   1st Qu.:-0.138   1st Qu.:-19.00  
##  Median : 0.020   Median : 0.000   Median :-0.025   Median :-13.00  
##  Mean   :-0.045   Mean   : 0.010   Mean   :-0.101   Mean   :-13.50  
##  3rd Qu.: 0.070   3rd Qu.: 0.020   3rd Qu.: 0.000   3rd Qu.: -8.75  
##  Max.   : 0.240   Max.   : 0.110   Max.   : 0.050   Max.   : 46.00  
##   accel_belt_y    accel_belt_z    magnet_belt_x   magnet_belt_y
##  Min.   :-16.0   Min.   :-187.0   Min.   :-13.0   Min.   :566  
##  1st Qu.:  2.0   1st Qu.: -24.0   1st Qu.:  5.5   1st Qu.:578  
##  Median :  4.5   Median :  27.0   Median : 33.5   Median :600  
##  Mean   : 18.4   Mean   : -17.6   Mean   : 35.1   Mean   :602  
##  3rd Qu.: 25.5   3rd Qu.:  38.2   3rd Qu.: 46.2   3rd Qu.:631  
##  Max.   : 72.0   Max.   :  49.0   Max.   :169.0   Max.   :638  
##  magnet_belt_z     roll_arm        pitch_arm         yaw_arm      
##  Min.   :-426   Min.   :-137.0   Min.   :-63.80   Min.   :-167.0  
##  1st Qu.:-398   1st Qu.:   0.0   1st Qu.: -9.19   1st Qu.: -60.1  
##  Median :-314   Median :   0.0   Median :  0.00   Median :   0.0  
##  Mean   :-347   Mean   :  16.4   Mean   : -3.95   Mean   :  -2.8  
##  3rd Qu.:-305   3rd Qu.:  71.5   3rd Qu.:  3.46   3rd Qu.:  25.5  
##  Max.   :-291   Max.   : 152.0   Max.   : 55.00   Max.   : 178.0  
##  total_accel_arm  gyros_arm_x      gyros_arm_y      gyros_arm_z    
##  Min.   : 3.0    Min.   :-3.710   Min.   :-2.090   Min.   :-0.690  
##  1st Qu.:20.2    1st Qu.:-0.645   1st Qu.:-0.635   1st Qu.:-0.180  
##  Median :29.5    Median : 0.020   Median :-0.040   Median :-0.025  
##  Mean   :26.4    Mean   : 0.077   Mean   :-0.160   Mean   : 0.120  
##  3rd Qu.:33.2    3rd Qu.: 1.248   3rd Qu.: 0.217   3rd Qu.: 0.565  
##  Max.   :44.0    Max.   : 3.660   Max.   : 1.850   Max.   : 1.130  
##   accel_arm_x      accel_arm_y     accel_arm_z      magnet_arm_x 
##  Min.   :-341.0   Min.   :-65.0   Min.   :-404.0   Min.   :-428  
##  1st Qu.:-277.0   1st Qu.: 52.2   1st Qu.:-128.5   1st Qu.:-374  
##  Median :-194.5   Median :112.0   Median : -83.5   Median :-265  
##  Mean   :-134.6   Mean   :103.1   Mean   : -87.8   Mean   : -39  
##  3rd Qu.:   5.5   3rd Qu.:168.2   3rd Qu.: -27.2   3rd Qu.: 250  
##  Max.   : 106.0   Max.   :245.0   Max.   :  93.0   Max.   : 750  
##   magnet_arm_y   magnet_arm_z  roll_dumbbell     pitch_dumbbell 
##  Min.   :-307   Min.   :-499   Min.   :-111.12   Min.   :-55.0  
##  1st Qu.: 205   1st Qu.: 403   1st Qu.:   7.49   1st Qu.:-51.9  
##  Median : 291   Median : 476   Median :  50.40   Median :-40.8  
##  Mean   : 239   Mean   : 370   Mean   :  33.76   Mean   :-19.5  
##  3rd Qu.: 359   3rd Qu.: 517   3rd Qu.:  58.13   3rd Qu.: 16.1  
##  Max.   : 474   Max.   : 633   Max.   : 123.98   Max.   : 96.9  
##   yaw_dumbbell     total_accel_dumbbell gyros_dumbbell_x gyros_dumbbell_y 
##  Min.   :-103.32   Min.   : 1.0         Min.   :-1.030   Min.   :-1.1100  
##  1st Qu.: -75.28   1st Qu.: 7.0         1st Qu.: 0.160   1st Qu.:-0.2100  
##  Median :  -8.29   Median :15.5         Median : 0.360   Median : 0.0150  
##  Mean   :  -0.94   Mean   :17.2         Mean   : 0.269   Mean   : 0.0605  
##  3rd Qu.:  55.83   3rd Qu.:29.0         3rd Qu.: 0.463   3rd Qu.: 0.1450  
##  Max.   : 132.23   Max.   :31.0         Max.   : 1.060   Max.   : 1.9100  
##  gyros_dumbbell_z accel_dumbbell_x accel_dumbbell_y accel_dumbbell_z
##  Min.   :-1.180   Min.   :-159.0   Min.   :-30.00   Min.   :-221.0  
##  1st Qu.:-0.485   1st Qu.:-140.2   1st Qu.:  5.75   1st Qu.:-192.2  
##  Median :-0.280   Median : -19.0   Median : 71.50   Median :  -3.0  
##  Mean   :-0.266   Mean   : -47.6   Mean   : 70.55   Mean   : -60.0  
##  3rd Qu.:-0.165   3rd Qu.:  15.8   3rd Qu.:151.25   3rd Qu.:  76.5  
##  Max.   : 1.100   Max.   : 185.0   Max.   :166.00   Max.   : 100.0  
##  magnet_dumbbell_x magnet_dumbbell_y magnet_dumbbell_z  roll_forearm   
##  Min.   :-576      Min.   :-558      Min.   :-164.0    Min.   :-176.0  
##  1st Qu.:-528      1st Qu.: 260      1st Qu.: -33.0    1st Qu.: -40.2  
##  Median :-508      Median : 316      Median :  49.5    Median :  94.2  
##  Mean   :-304      Mean   : 189      Mean   :  71.4    Mean   :  38.7  
##  3rd Qu.:-317      3rd Qu.: 348      3rd Qu.:  96.2    3rd Qu.: 143.2  
##  Max.   : 523      Max.   : 403      Max.   : 368.0    Max.   : 176.0  
##  pitch_forearm     yaw_forearm      total_accel_forearm gyros_forearm_x 
##  Min.   :-63.50   Min.   :-168.00   Min.   :21.0        Min.   :-1.060  
##  1st Qu.:-11.46   1st Qu.: -93.38   1st Qu.:24.0        1st Qu.:-0.585  
##  Median :  8.83   Median : -19.25   Median :32.5        Median : 0.020  
##  Mean   :  7.10   Mean   :   2.19   Mean   :32.0        Mean   :-0.020  
##  3rd Qu.: 28.50   3rd Qu.: 104.50   3rd Qu.:36.8        3rd Qu.: 0.292  
##  Max.   : 59.30   Max.   : 159.00   Max.   :47.0        Max.   : 1.380  
##  gyros_forearm_y  gyros_forearm_z   accel_forearm_x  accel_forearm_y 
##  Min.   :-5.970   Min.   :-1.2600   Min.   :-212.0   Min.   :-331.0  
##  1st Qu.:-1.288   1st Qu.:-0.0975   1st Qu.:-114.8   1st Qu.:   8.5  
##  Median : 0.035   Median : 0.2300   Median :  86.0   Median : 138.0  
##  Mean   :-0.042   Mean   : 0.2610   Mean   :  38.8   Mean   : 125.3  
##  3rd Qu.: 2.047   3rd Qu.: 0.7625   3rd Qu.: 166.2   3rd Qu.: 268.0  
##  Max.   : 4.260   Max.   : 1.8000   Max.   : 232.0   Max.   : 406.0  
##  accel_forearm_z  magnet_forearm_x magnet_forearm_y magnet_forearm_z
##  Min.   :-282.0   Min.   :-714.0   Min.   :-787     Min.   :-32     
##  1st Qu.:-199.0   1st Qu.:-427.2   1st Qu.:-329     1st Qu.:275     
##  Median :-148.5   Median :-189.5   Median : 487     Median :492     
##  Mean   : -93.7   Mean   :-159.2   Mean   : 192     Mean   :460     
##  3rd Qu.: -31.0   3rd Qu.:  41.5   3rd Qu.: 721     3rd Qu.:662     
##  Max.   : 179.0   Max.   : 532.0   Max.   : 800     Max.   :884     
##    problem_id   
##  Min.   : 1.00  
##  1st Qu.: 5.75  
##  Median :10.50  
##  Mean   :10.50  
##  3rd Qu.:15.25  
##  Max.   :20.00
```

####VIII.2 Cross Validation Results####

```r
print(predcv)
```

```
##     3     4     5     6     8    10    15    18    23    25    26    31 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##    33    34    36    46    54    55    57    61    62    63    64    70 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##    71    72    73    74    75    78    81    82    84    96   100   105 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   107   109   112   114   115   116   121   122   123   128   132   133 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   135   142   145   147   150   161   166   167   169   171   173   189 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   190   191   195   197   200   202   203   209   210   215   216   217 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   219   223   240   241   242   244   251   258   261   265   266   270 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   275   277   279   280   282   285   286   291   295   296   299   304 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   307   308   309   311   316   320   323   328   331   333   334   335 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   337   340   341   345   346   351   358   361   368   369   376   377 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   380   394   395   396   408   409   410   417   423   429   431   432 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   434   435   436   443   450   451   454   455   462   463   467   470 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   472   473   477   480   487   491   493   496   499   506   511   512 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   520   524   526   530   532   535   537   538   539   541   551   560 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   561   564   565   566   570   572   576   577   578   588   590   596 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   598   601   603   605   607   608   609   615   622   625   633   638 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   643   650   651   653   654   656   658   660   662   663   664   665 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   669   670   671   672   675   676   683   689   690   693   697   699 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   700   705   708   712   716   717   718   722   724   726   728   730 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   732   734   738   742   748   756   762   763   764   766   770   775 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   780   782   787   789   790   791   799   804   811   812   816   818 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   826   834   836   838   852   857   861   865   867   869   874   878 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   884   887   891   893   894   898   900   902   906   907   912   913 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   914   916   918   921   927   928   931   932   942   943   944   946 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   947   951   953   955   956   958   963   964   967   973   981   993 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##   994   997  1000  1002  1006  1010  1015  1016  1017  1018  1019  1020 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1023  1025  1027  1031  1036  1039  1041  1046  1048  1051  1061  1065 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1067  1071  1074  1075  1082  1083  1095  1097  1099  1102  1103  1104 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1108  1109  1117  1123  1124  1128  1131  1133  1140  1143  1147  1149 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1151  1154  1155  1157  1165  1167  1168  1170  1172  1177  1178  1179 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1182  1183  1188  1192  1194  1195  1197  1198  1199  1204  1206  1209 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1220  1221  1223  1225  1226  1227  1237  1240  1242  1245  1254  1256 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1257  1259  1260  1270  1271  1274  1280  1282  1283  1284  1286  1288 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1293  1307  1308  1310  1311  1322  1324  1329  1330  1332  1334  1345 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1346  1347  1350  1351  1352  1358  1364  1365  1366  1375  1380  1382 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1384  1385  1386  1388  1390  1392  1395  1396  1398  1405  1407  1410 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1413  1415  1418  1421  1423  1425  1426  1428  1429  1430  1433  1436 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1441  1442  1443  1444  1449  1453  1461  1463  1466  1472  1477  1480 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1481  1483  1486  1489  1490  1497  1511  1515  1516  1517  1518  1520 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1521  1524  1526  1527  1532  1537  1538  1540  1544  1546  1549  1550 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1551  1561  1562  1570  1581  1582  1584  1586  1588  1590  1593  1594 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1595  1599  1600  1602  1605  1611  1612  1614  1617  1619  1620  1644 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1645  1646  1654  1659  1664  1669  1681  1682  1687  1690  1699  1704 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1708  1713  1721  1722  1723  1727  1732  1733  1739  1740  1742  1743 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1746  1757  1758  1767  1770  1780  1782  1784  1787  1791  1792  1793 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1796  1797  1801  1802  1807  1809  1816  1817  1820  1822  1823  1824 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1834  1835  1837  1839  1840  1844  1849  1851  1852  1855  1859  1860 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1861  1862  1865  1868  1883  1884  1888  1890  1892  1893  1894  1903 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1906  1907  1912  1913  1917  1921  1924  1925  1926  1929  1936  1938 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1940  1951  1958  1965  1967  1968  1969  1970  1971  1975  1980  1983 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  1984  1987  1990  1991  1992  1994  1996  2009  2011  2012  2024  2025 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2026  2027  2034  2035  2037  2038  2043  2044  2049  2052  2053  2062 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2063  2068  2077  2078  2079  2086  2087  2088  2094  2097  2102  2111 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2114  2115  2120  2123  2127  2134  2136  2139  2140  2141  2142  2143 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2144  2147  2148  2162  2164  2165  2168  2170  2172  2175  2179  2187 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2189  2190  2192  2196  2201  2211  2212  2224  2230  2244  2248  2250 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2254  2257  2261  2262  2264  2267  2268  2279  2280  2290  2296  2300 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2301  2302  2307  2308  2309  2313  2314  2317  2318  2321  2322  2327 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2329  2330  2334  2335  2338  2340  2346  2357  2360  2362  2363  2364 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2366  2368  2369  2371  2372  2373  2374  2378  2380  2381  2383  2388 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2394  2401  2408  2410  2413  2414  2416  2417  2419  2420  2424  2427 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2432  2436  2440  2444  2445  2455  2456  2457  2459  2461  2462  2464 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2466  2468  2472  2476  2482  2484  2489  2495  2499  2500  2501  2503 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2506  2513  2518  2521  2522  2523  2524  2529  2532  2542  2544  2545 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2546  2548  2550  2555  2557  2560  2574  2575  2578  2583  2584  2585 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2588  2589  2600  2601  2603  2607  2614  2617  2619  2621  2623  2629 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2631  2632  2634  2636  2637  2643  2646  2647  2649  2652  2656  2658 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2664  2668  2670  2672  2674  2677  2679  2686  2689  2698  2700  2701 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2710  2717  2718  2724  2725  2726  2729  2731  2733  2741  2745  2752 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2753  2756  2764  2767  2769  2772  2773  2776  2784  2787  2788  2792 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2793  2799  2801  2805  2807  2811  2814  2817  2818  2820  2822  2826 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2831  2832  2836  2840  2841  2843  2847  2848  2850  2851  2855  2858 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2860  2864  2868  2869  2878  2884  2890  2891  2892  2893  2894  2898 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2899  2907  2909  2912  2914  2918  2920  2922  2924  2929  2936  2949 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2951  2953  2955  2956  2963  2964  2966  2970  2973  2974  2976  2979 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  2981  2982  2989  2992  2997  3000  3003  3007  3010  3015  3021  3024 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3033  3035  3042  3046  3048  3049  3051  3056  3059  3063  3068  3070 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3071  3074  3078  3081  3093  3096  3099  3102  3106  3110  3112  3117 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3123  3125  3126  3129  3130  3131  3134  3139  3142  3143  3146  3147 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3149  3155  3164  3166  3171  3173  3174  3176  3189  3195  3202  3209 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3210  3218  3223  3224  3225  3226  3228  3235  3236  3237  3247  3248 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3251  3252  3254  3255  3259  3260  3262  3263  3264  3267  3268  3284 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3285  3286  3287  3295  3296  3298  3299  3302  3304  3310  3313  3314 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3315  3318  3325  3328  3332  3338  3341  3344  3346  3347  3354  3365 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3367  3368  3370  3373  3376  3379  3380  3384  3389  3394  3402  3419 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3421  3424  3426  3430  3442  3447  3457  3458  3459  3461  3466  3469 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3472  3477  3480  3481  3483  3484  3487  3491  3498  3499  3500  3501 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3507  3509  3510  3512  3515  3519  3520  3528  3532  3544  3545  3546 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3547  3559  3569  3577  3579  3582  3586  3588  3596  3602  3608  3609 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3615  3618  3619  3623  3641  3642  3643  3645  3646  3647  3651  3652 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3655  3657  3660  3665  3673  3674  3675  3676  3677  3679  3682  3683 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3695  3697  3704  3713  3715  3717  3718  3719  3724  3729  3735  3736 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3737  3738  3739  3740  3744  3746  3754  3758  3766  3767  3772  3775 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3776  3780  3787  3792  3793  3795  3798  3799  3803  3806  3807  3808 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3810  3814  3815  3817  3827  3831  3833  3835  3843  3845  3847  3854 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3856  3857  3861  3862  3863  3867  3868  3874  3876  3880  3881  3884 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3888  3889  3890  3891  3897  3899  3902  3906  3907  3910  3916  3917 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3918  3924  3926  3929  3930  3932  3936  3937  3938  3940  3942  3946 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3949  3953  3955  3960  3962  3964  3970  3971  3975  3976  3978  3980 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  3981  3989  3991  3992  3993  4002  4004  4007  4011  4013  4014  4019 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4024  4026  4029  4030  4036  4038  4039  4041  4042  4043  4045  4046 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4047  4050  4052  4053  4060  4062  4066  4072  4073  4075  4078  4086 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4087  4088  4092  4099  4102  4103  4104  4112  4116  4119  4124  4125 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4131  4132  4133  4137  4138  4140  4142  4148  4149  4158  4160  4163 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4172  4174  4176  4178  4179  4183  4185  4186  4192  4201  4204  4205 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4208  4210  4211  4212  4215  4217  4220  4228  4230  4237  4239  4243 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4244  4252  4254  4255  4256  4257  4258  4263  4265  4272  4274  4276 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4279  4280  4287  4289  4291  4293  4298  4299  4300  4307  4309  4312 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4315  4316  4317  4322  4324  4325  4332  4340  4343  4344  4345  4347 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4352  4355  4356  4361  4363  4365  4372  4375  4376  4378  4379  4380 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4382  4384  4387  4388  4393  4395  4403  4404  4411  4412  4414  4418 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4422  4423  4424  4427  4439  4441  4442  4443  4448  4450  4451  4455 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4460  4464  4468  4474  4477  4479  4481  4482  4484  4485  4486  4489 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4492  4493  4497  4504  4506  4507  4508  4512  4514  4521  4527  4531 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4534  4537  4538  4544  4545  4546  4556  4561  4563  4565  4572  4574 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4578  4582  4586  4591  4593  4595  4600  4606  4607  4608  4609  4610 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4613  4616  4620  4627  4637  4640  4641  4646  4647  4650  4660  4663 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4665  4676  4677  4683  4693  4700  4704  4706  4707  4709  4711  4718 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4719  4724  4725  4726  4727  4731  4732  4733  4736  4747  4750  4755 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4757  4760  4762  4767  4777  4780  4787  4789  4790  4791  4795  4798 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4805  4808  4809  4810  4814  4816  4820  4823  4830  4833  4840  4841 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4843  4847  4856  4859  4861  4871  4881  4883  4888  4890  4897  4899 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4900  4903  4904  4908  4910  4912  4913  4924  4926  4928  4930  4938 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4940  4944  4945  4946  4948  4950  4959  4965  4967  4969  4971  4972 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  4973  4974  4978  4993  4994  4995  4997  5001  5002  5004  5005  5008 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5017  5023  5025  5028  5029  5030  5031  5032  5033  5035  5042  5043 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5045  5051  5052  5054  5055  5058  5063  5070  5075  5077  5082  5083 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5084  5086  5094  5095  5100  5104  5108  5111  5114  5117  5119  5120 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5122  5123  5128  5130  5134  5135  5136  5141  5142  5143  5148  5152 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5153  5157  5159  5161  5163  5165  5167  5177  5178  5181  5183  5188 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5189  5190  5191  5193  5195  5196  5198  5199  5208  5212  5213  5217 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5218  5221  5226  5231  5233  5235  5243  5249  5255  5263  5269  5272 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5274  5276  5280  5285  5287  5296  5297  5299  5306  5310  5311  5314 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5317  5318  5319  5326  5327  5335  5336  5338  5339  5341  5354  5356 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5357  5364  5371  5375  5376  5378  5380  5381  5385  5390  5393  5394 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5396  5397  5408  5411  5412  5419  5422  5424  5428  5429  5439  5450 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5453  5455  5458  5460  5462  5463  5464  5469  5471  5480  5486  5488 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5489  5490  5494  5502  5504  5507  5508  5513  5515  5517  5521  5524 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5528  5531  5538  5539  5542  5544  5547  5550  5554  5557  5560  5561 
##     A     A     A     A     A     A     A     A     A     A     A     A 
##  5563  5568  5569  5571  5573  5577  5586  5587  5588  5589  5590  5591 
##     A     A     A     A     A     A     B     A     A     A     B     B 
##  5596  5598  5599  5603  5605  5609  5610  5613  5614  5618  5620  5621 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  5622  5624  5626  5627  5630  5637  5645  5654  5655  5656  5661  5663 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  5665  5666  5669  5672  5678  5679  5680  5681  5683  5691  5694  5697 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  5700  5703  5707  5709  5712  5716  5717  5718  5720  5728  5730  5742 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  5744  5746  5748  5749  5750  5755  5758  5760  5764  5767  5768  5772 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  5774  5779  5783  5790  5796  5798  5799  5802  5806  5807  5810  5816 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  5823  5826  5829  5830  5834  5837  5838  5841  5845  5871  5876  5879 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  5881  5883  5901  5903  5905  5906  5907  5911  5913  5917  5919  5920 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  5921  5922  5932  5933  5936  5943  5948  5949  5951  5959  5961  5964 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  5967  5969  5970  5972  5973  5974  5978  5981  5984  5988  5992  5993 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  5996  5999  6000  6001  6004  6008  6009  6013  6014  6016  6017  6018 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6022  6030  6032  6036  6037  6038  6041  6047  6049  6051  6059  6063 
##     A     B     B     B     B     B     B     B     B     B     B     B 
##  6065  6070  6071  6072  6076  6081  6082  6088  6089  6092  6093  6095 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6105  6110  6114  6116  6118  6120  6126  6129  6134  6146  6148  6149 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6151  6155  6158  6164  6165  6168  6180  6182  6183  6184  6186  6188 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6194  6195  6196  6197  6199  6200  6202  6206  6207  6209  6210  6211 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6213  6215  6216  6217  6221  6224  6226  6227  6238  6240  6244  6245 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6249  6251  6253  6254  6260  6262  6263  6266  6267  6269  6270  6272 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6275  6278  6279  6282  6285  6286  6291  6294  6298  6299  6303  6308 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6309  6314  6317  6327  6329  6330  6333  6338  6346  6348  6353  6356 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6357  6359  6360  6361  6366  6367  6369  6370  6379  6381  6385  6387 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6389  6391  6392  6394  6406  6407  6409  6411  6412  6413  6414  6421 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6425  6427  6428  6429  6433  6434  6438  6440  6442  6443  6445  6452 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6457  6458  6463  6465  6466  6471  6473  6474  6483  6484  6488  6490 
##     B     B     B     B     B     B     A     A     B     B     B     B 
##  6493  6506  6507  6510  6512  6514  6526  6528  6529  6533  6537  6539 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6542  6546  6548  6551  6581  6588  6591  6593  6595  6596  6597  6602 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6603  6604  6609  6613  6616  6617  6621  6631  6632  6638  6643  6644 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6645  6646  6650  6655  6667  6669  6672  6674  6681  6685  6687  6689 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6690  6691  6694  6699  6701  6703  6707  6709  6712  6714  6721  6729 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6730  6733  6738  6744  6746  6750  6751  6763  6767  6768  6772  6774 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6776  6780  6784  6789  6793  6794  6800  6807  6821  6830  6834  6835 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6845  6851  6852  6855  6859  6860  6861  6868  6873  6877  6880  6882 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6884  6887  6888  6889  6890  6891  6892  6896  6899  6900  6902  6910 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6912  6919  6923  6928  6930  6940  6942  6943  6946  6953  6956  6958 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6965  6968  6971  6974  6975  6983  6984  6985  6987  6988  6991  6993 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  6994  6997  6998  7000  7006  7009  7016  7023  7024  7030  7033  7048 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7051  7053  7054  7055  7060  7064  7065  7067  7069  7070  7072  7079 
##     B     B     B     B     B     B     B     B     B     B     B     C 
##  7081  7083  7092  7095  7104  7105  7107  7108  7111  7114  7116  7127 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7134  7135  7137  7143  7150  7151  7160  7161  7163  7164  7171  7177 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7179  7183  7193  7197  7198  7199  7203  7204  7209  7220  7221  7224 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7229  7233  7234  7237  7238  7243  7248  7251  7257  7265  7268  7269 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7275  7282  7283  7284  7286  7293  7297  7298  7299  7306  7307  7309 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7313  7321  7325  7326  7327  7331  7337  7338  7339  7341  7348  7349 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7354  7358  7362  7363  7364  7369  7372  7376  7379  7381  7382  7388 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7390  7394  7396  7400  7406  7417  7418  7420  7423  7427  7429  7431 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7447  7449  7450  7451  7452  7457  7459  7470  7471  7476  7479  7491 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7492  7493  7494  7502  7503  7509  7510  7516  7522  7525  7526  7533 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7536  7540  7544  7556  7560  7561  7562  7563  7565  7566  7569  7570 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7572  7576  7583  7584  7587  7590  7592  7598  7606  7607  7611  7613 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7617  7627  7631  7635  7640  7642  7650  7652  7653  7655  7659  7665 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7667  7668  7672  7675  7676  7678  7684  7687  7688  7693  7699  7700 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7705  7709  7710  7711  7715  7716  7718  7721  7722  7726  7727  7729 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7731  7732  7734  7736  7738  7743  7749  7750  7754  7758  7762  7767 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7770  7773  7774  7781  7782  7783  7787  7788  7803  7804  7805  7806 
##     B     B     B     B     B     B     B     C     B     B     B     B 
##  7807  7810  7813  7820  7824  7826  7829  7830  7837  7840  7842  7844 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7846  7847  7849  7851  7854  7857  7860  7863  7870  7873  7886  7888 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7891  7892  7895  7896  7902  7904  7907  7909  7910  7912  7913  7922 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7923  7924  7932  7934  7936  7948  7949  7951  7953  7954  7960  7964 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  7971  7976  7977  7980  7985  7988  7994  7997  8000  8001  8004  8005 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8014  8024  8028  8029  8037  8038  8039  8044  8047  8050  8051  8052 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8056  8058  8061  8065  8066  8075  8076  8077  8081  8083  8085  8087 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8089  8091  8092  8093  8094  8100  8105  8106  8109  8115  8125  8128 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8132  8139  8142  8144  8148  8152  8160  8163  8166  8170  8173  8176 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8178  8181  8184  8186  8188  8190  8193  8197  8206  8208  8218  8239 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8240  8247  8248  8258  8261  8263  8265  8266  8271  8273  8278  8283 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8284  8285  8288  8290  8292  8295  8299  8304  8306  8307  8309  8317 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8320  8324  8334  8335  8336  8337  8339  8340  8344  8349  8352  8354 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8355  8356  8359  8365  8368  8370  8373  8376  8377  8380  8383  8386 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8392  8396  8400  8401  8404  8406  8410  8418  8423  8426  8432  8434 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8435  8437  8439  8441  8442  8447  8448  8453  8454  8455  8462  8464 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8467  8468  8470  8472  8474  8484  8485  8490  8493  8495  8502  8520 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8524  8528  8529  8530  8531  8532  8538  8540  8542  8545  8553  8555 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8556  8557  8558  8559  8561  8563  8567  8568  8573  8574  8575  8577 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8581  8585  8586  8587  8591  8595  8596  8599  8602  8603  8605  8607 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8608  8611  8613  8619  8626  8627  8632  8633  8637  8638  8640  8641 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8646  8653  8657  8659  8663  8665  8673  8677  8678  8683  8692  8693 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8694  8698  8704  8710  8711  8712  8715  8716  8717  8719  8723  8726 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8727  8730  8731  8732  8735  8738  8740  8743  8744  8752  8756  8766 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8768  8769  8771  8772  8773  8775  8777  8778  8781  8786  8787  8789 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8790  8792  8794  8799  8805  8807  8808  8809  8816  8825  8828  8832 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8833  8839  8840  8841  8843  8845  8846  8847  8851  8852  8857  8858 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8864  8868  8876  8877  8879  8880  8882  8884  8893  8897  8899  8903 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8904  8906  8933  8934  8935  8941  8945  8946  8948  8950  8951  8955 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8957  8958  8961  8964  8966  8972  8974  8977  8984  8985  8986  8987 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  8990  8992  8993  9001  9002  9003  9004  9011  9019  9020  9022  9024 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  9029  9030  9036  9037  9038  9053  9058  9060  9068  9070  9076  9080 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  9081  9083  9087  9089  9094  9095  9096  9105  9106  9110  9112  9115 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  9117  9120  9121  9124  9125  9126  9136  9137  9138  9141  9152  9166 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  9168  9170  9172  9177  9179  9183  9184  9185  9186  9191  9192  9196 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  9197  9198  9200  9201  9202  9208  9210  9211  9216  9218  9223  9224 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  9226  9229  9232  9233  9235  9237  9238  9251  9254  9256  9259  9260 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  9269  9272  9273  9274  9275  9277  9278  9281  9284  9288  9292  9294 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  9296  9297  9298  9299  9303  9305  9307  9309  9313  9315  9316  9318 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  9320  9321  9322  9324  9331  9332  9335  9338  9344  9354  9359  9360 
##     B     B     B     B     B     B     B     B     B     B     B     B 
##  9363  9365  9372  9373  9375  9379  9384  9385  9386  9388  9389  9392 
##     B     B     B     B     B     C     C     C     C     C     C     C 
##  9396  9397  9403  9405  9406  9407  9408  9410  9413  9414  9417  9418 
##     C     C     B     C     C     C     C     C     C     C     B     C 
##  9430  9433  9436  9437  9439  9443  9444  9450  9452  9457  9459  9460 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9465  9466  9467  9468  9472  9475  9478  9479  9482  9489  9491  9492 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9499  9502  9505  9506  9508  9510  9514  9524  9525  9528  9529  9533 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9535  9537  9538  9539  9540  9545  9547  9548  9551  9554  9556  9560 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9566  9567  9570  9582  9585  9587  9591  9598  9603  9609  9612  9620 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9626  9633  9640  9641  9643  9644  9648  9651  9654  9655  9659  9660 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9661  9663  9672  9676  9677  9681  9683  9684  9688  9691  9700  9704 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9709  9713  9715  9717  9720  9723  9724  9725  9727  9728  9733  9736 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9741  9745  9747  9761  9763  9766  9767  9770  9772  9773  9776  9778 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9779  9781  9784  9785  9788  9791  9793  9796  9800  9802  9805  9811 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9815  9816  9819  9828  9831  9832  9834  9838  9841  9845  9849  9853 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9859  9863  9865  9871  9874  9876  9877  9878  9881  9888  9889  9890 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9892  9896  9898  9899  9902  9903  9907  9916  9917  9929  9931  9934 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9935  9936  9938  9940  9941  9961  9967  9968  9978  9980  9983  9993 
##     C     C     C     C     C     C     C     C     C     C     C     C 
##  9998 10000 10010 10012 10016 10017 10020 10023 10027 10042 10043 10046 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10048 10050 10052 10055 10059 10063 10071 10075 10079 10087 10089 10094 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10101 10103 10105 10108 10113 10117 10123 10124 10134 10146 10147 10148 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10149 10150 10151 10153 10168 10174 10177 10178 10179 10182 10183 10192 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10193 10200 10201 10202 10204 10207 10208 10213 10216 10217 10219 10221 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10223 10224 10229 10230 10233 10235 10236 10237 10238 10239 10240 10241 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10243 10244 10248 10249 10252 10256 10260 10261 10268 10273 10274 10281 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10283 10285 10287 10290 10295 10296 10297 10298 10303 10307 10309 10311 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10312 10313 10314 10319 10320 10325 10330 10331 10332 10336 10337 10338 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10345 10346 10347 10350 10351 10353 10354 10355 10357 10358 10362 10366 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10370 10375 10383 10385 10392 10396 10399 10408 10415 10416 10419 10423 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10427 10428 10430 10434 10436 10437 10439 10441 10442 10452 10456 10459 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10460 10464 10467 10470 10472 10476 10477 10479 10482 10487 10495 10501 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10507 10511 10512 10513 10515 10516 10518 10523 10526 10538 10541 10545 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10546 10547 10548 10555 10561 10568 10569 10572 10574 10576 10578 10579 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10581 10587 10595 10602 10603 10610 10611 10612 10613 10618 10619 10621 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10630 10632 10635 10636 10638 10646 10650 10652 10668 10671 10673 10674 
##     C     B     C     C     C     C     C     C     C     C     C     C 
## 10681 10682 10684 10692 10698 10700 10701 10704 10705 10709 10719 10723 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10724 10733 10735 10736 10738 10739 10741 10746 10751 10754 10755 10757 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10761 10766 10767 10776 10778 10779 10789 10791 10792 10793 10797 10814 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10819 10826 10827 10829 10830 10833 10845 10847 10855 10856 10857 10861 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10866 10868 10872 10879 10880 10884 10886 10887 10890 10891 10894 10895 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10896 10904 10905 10907 10910 10912 10917 10923 10926 10930 10931 10938 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10939 10940 10943 10945 10946 10950 10951 10955 10956 10957 10958 10965 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 10983 10985 10990 10991 10992 10994 11003 11005 11006 11012 11013 11015 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11016 11019 11023 11024 11027 11028 11029 11030 11034 11039 11050 11055 
##     C     C     C     C     C     C     C     C     C     C     C     D 
## 11057 11063 11069 11072 11074 11077 11078 11083 11086 11087 11094 11095 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11096 11098 11104 11105 11107 11110 11112 11116 11117 11120 11122 11125 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11126 11129 11132 11138 11139 11142 11146 11147 11152 11155 11160 11164 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11166 11170 11173 11175 11178 11180 11184 11185 11188 11194 11195 11198 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11200 11206 11208 11209 11210 11214 11215 11217 11223 11224 11225 11227 
##     C     C     B     B     B     C     C     C     C     C     C     C 
## 11230 11231 11234 11235 11237 11242 11248 11250 11251 11254 11258 11259 
##     C     C     C     B     C     C     C     C     C     C     C     C 
## 11261 11266 11267 11269 11271 11274 11276 11278 11282 11284 11294 11295 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11296 11298 11299 11302 11306 11308 11314 11318 11320 11322 11332 11333 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11336 11345 11346 11350 11356 11361 11363 11366 11368 11369 11371 11374 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11375 11386 11387 11388 11389 11391 11392 11401 11407 11408 11409 11412 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11413 11417 11418 11420 11427 11429 11434 11437 11438 11441 11443 11444 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11447 11449 11451 11456 11460 11461 11463 11466 11467 11469 11471 11472 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11473 11476 11479 11481 11482 11483 11489 11492 11493 11494 11495 11496 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11497 11499 11504 11508 11510 11514 11517 11522 11523 11524 11525 11531 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11532 11536 11539 11544 11545 11548 11549 11550 11553 11555 11562 11566 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11568 11569 11571 11572 11574 11578 11580 11581 11592 11599 11600 11607 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11610 11613 11620 11624 11628 11629 11632 11635 11636 11649 11652 11658 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11673 11675 11677 11678 11679 11680 11684 11687 11688 11694 11696 11700 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11701 11709 11711 11715 11721 11724 11725 11726 11729 11733 11743 11744 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11746 11753 11755 11758 11760 11762 11766 11767 11770 11772 11773 11783 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11785 11787 11795 11798 11801 11818 11820 11823 11828 11831 11836 11837 
##     C     C     C     C     C     C     C     D     C     C     C     C 
## 11840 11848 11849 11852 11853 11865 11867 11872 11877 11879 11880 11882 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11884 11885 11886 11903 11906 11911 11913 11921 11922 11923 11929 11932 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 11938 11943 11948 11952 11953 11957 11959 11962 11967 11973 11975 11980 
##     C     C     C     C     C     C     B     C     C     C     C     C 
## 11983 11991 11997 12002 12004 12006 12008 12010 12011 12019 12022 12029 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12030 12032 12033 12045 12049 12052 12058 12061 12066 12067 12071 12072 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12075 12077 12078 12082 12084 12086 12090 12093 12094 12106 12107 12109 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12114 12117 12124 12127 12130 12136 12138 12140 12142 12143 12145 12148 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12149 12157 12177 12178 12179 12181 12184 12185 12189 12192 12193 12195 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12196 12198 12205 12207 12210 12219 12220 12224 12227 12229 12235 12236 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12245 12246 12247 12257 12272 12279 12285 12286 12292 12297 12304 12306 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12315 12316 12319 12320 12324 12325 12326 12328 12334 12335 12336 12339 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12341 12343 12347 12348 12349 12351 12355 12356 12360 12364 12365 12369 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12381 12384 12385 12388 12393 12395 12396 12398 12400 12401 12410 12413 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12414 12417 12424 12425 12434 12436 12437 12440 12444 12445 12452 12454 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12456 12458 12459 12460 12461 12465 12474 12476 12479 12482 12484 12488 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12490 12506 12509 12521 12523 12524 12525 12535 12539 12540 12541 12547 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12549 12551 12553 12560 12564 12565 12571 12574 12579 12586 12589 12593 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12594 12595 12598 12601 12604 12605 12607 12608 12609 12610 12613 12616 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12621 12622 12625 12628 12629 12630 12631 12638 12639 12642 12643 12649 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12654 12660 12664 12667 12670 12673 12676 12679 12681 12682 12690 12692 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12693 12704 12706 12709 12711 12715 12717 12718 12723 12726 12727 12742 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12745 12746 12748 12749 12750 12751 12754 12755 12762 12765 12767 12769 
##     C     C     C     C     C     C     C     C     C     C     C     C 
## 12775 12776 12778 12780 12781 12782 12783 12786 12793 12794 12798 12801 
##     C     C     C     C     C     C     C     C     C     C     C     D 
## 12802 12806 12807 12814 12817 12819 12824 12831 12835 12837 12839 12844 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 12853 12854 12863 12867 12876 12879 12882 12883 12887 12889 12891 12893 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 12894 12898 12899 12903 12905 12906 12908 12912 12915 12916 12918 12919 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 12920 12922 12923 12926 12927 12930 12934 12936 12937 12940 12941 12942 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 12945 12948 12949 12950 12953 12957 12958 12961 12963 12966 12967 12968 
##     D     C     C     D     D     D     D     C     D     C     C     C 
## 12969 12971 12972 12976 12982 12986 12990 12992 12996 13002 13003 13004 
##     C     D     D     D     D     D     D     D     D     D     D     D 
## 13008 13011 13012 13015 13022 13026 13034 13035 13039 13044 13049 13052 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13054 13058 13061 13062 13064 13068 13074 13076 13080 13087 13088 13089 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13092 13094 13095 13098 13100 13101 13104 13105 13106 13110 13111 13122 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13128 13129 13132 13133 13134 13136 13139 13143 13150 13153 13154 13155 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13159 13163 13169 13172 13173 13175 13176 13180 13181 13182 13188 13191 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13194 13196 13197 13204 13205 13210 13219 13225 13226 13230 13231 13241 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13244 13245 13253 13255 13266 13267 13268 13270 13271 13272 13274 13275 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13277 13281 13282 13284 13293 13296 13297 13303 13304 13306 13311 13313 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13315 13316 13319 13320 13323 13324 13325 13331 13332 13333 13345 13347 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13352 13357 13358 13359 13360 13365 13368 13371 13373 13378 13387 13389 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13397 13400 13409 13412 13419 13422 13426 13427 13429 13437 13441 13443 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13445 13453 13455 13458 13462 13466 13472 13475 13476 13479 13483 13484 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13496 13503 13510 13512 13513 13518 13519 13522 13526 13533 13538 13542 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13543 13544 13545 13548 13551 13552 13553 13555 13558 13560 13562 13567 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13569 13570 13571 13580 13588 13591 13594 13595 13597 13606 13607 13610 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13615 13618 13619 13620 13622 13629 13630 13644 13647 13648 13653 13656 
##     D     D     D     D     D     E     C     D     D     D     D     D 
## 13669 13672 13674 13675 13678 13680 13686 13691 13695 13702 13703 13713 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13716 13725 13727 13729 13730 13732 13740 13742 13744 13745 13748 13754 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13760 13766 13770 13772 13776 13782 13793 13794 13795 13796 13799 13800 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13801 13810 13819 13827 13831 13833 13837 13838 13846 13848 13854 13857 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13858 13862 13863 13867 13869 13870 13872 13875 13877 13878 13879 13891 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13894 13901 13902 13906 13910 13912 13919 13921 13923 13924 13929 13935 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13936 13937 13940 13942 13943 13949 13952 13956 13959 13960 13962 13965 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 13967 13971 13977 13979 13981 13983 13986 13994 13998 14000 14002 14006 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14008 14009 14010 14013 14017 14022 14025 14035 14036 14038 14039 14050 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14056 14063 14064 14065 14069 14074 14078 14080 14088 14090 14094 14095 
##     D     D     D     D     D     D     D     D     D     C     C     C 
## 14096 14099 14101 14102 14105 14107 14111 14113 14116 14117 14122 14125 
##     C     D     D     D     D     D     D     D     D     D     D     D 
## 14129 14133 14135 14140 14145 14149 14153 14156 14165 14174 14178 14179 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14181 14182 14183 14184 14185 14188 14190 14195 14201 14202 14203 14205 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14212 14214 14215 14219 14221 14222 14223 14224 14225 14226 14232 14233 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14234 14235 14238 14250 14252 14255 14257 14264 14266 14269 14271 14273 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14285 14288 14290 14292 14293 14294 14297 14299 14300 14301 14304 14305 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14307 14312 14314 14315 14320 14322 14329 14332 14336 14337 14341 14342 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14343 14345 14346 14355 14356 14357 14361 14365 14367 14368 14370 14371 
##     D     D     D     D     D     D     D     D     D     D     C     C 
## 14372 14374 14377 14381 14382 14383 14384 14385 14387 14389 14390 14395 
##     C     D     D     D     D     D     D     D     D     C     D     D 
## 14396 14397 14402 14407 14410 14418 14421 14422 14425 14426 14427 14428 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14431 14439 14440 14446 14447 14448 14457 14458 14461 14468 14472 14476 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14484 14493 14494 14495 14502 14506 14510 14511 14515 14516 14521 14522 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14526 14528 14532 14536 14541 14550 14553 14555 14556 14559 14560 14561 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14565 14569 14571 14575 14590 14595 14600 14606 14609 14610 14611 14614 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14619 14621 14624 14625 14626 14627 14631 14633 14636 14638 14655 14656 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14662 14664 14666 14667 14668 14672 14678 14683 14688 14693 14700 14702 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14703 14712 14714 14718 14721 14726 14728 14729 14730 14735 14738 14739 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14742 14747 14748 14749 14750 14752 14754 14757 14758 14762 14765 14772 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14793 14797 14804 14806 14813 14817 14823 14828 14831 14838 14843 14845 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14847 14851 14855 14871 14873 14876 14879 14880 14884 14885 14899 14912 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14926 14928 14933 14934 14935 14938 14943 14945 14950 14951 14952 14955 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14956 14958 14963 14965 14967 14969 14971 14974 14977 14978 14980 14981 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 14982 14988 14993 15001 15002 15003 15007 15009 15010 15017 15018 15019 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15024 15027 15032 15034 15038 15043 15045 15047 15053 15056 15057 15060 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15061 15063 15065 15070 15072 15074 15076 15077 15079 15080 15084 15091 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15092 15097 15098 15101 15102 15103 15104 15106 15110 15118 15122 15126 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15129 15130 15135 15138 15141 15145 15149 15150 15152 15153 15154 15155 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15158 15160 15161 15162 15163 15165 15169 15170 15174 15178 15179 15180 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15181 15187 15189 15192 15193 15195 15202 15205 15207 15212 15215 15219 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15221 15224 15227 15233 15235 15236 15240 15241 15252 15260 15263 15264 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15268 15269 15271 15272 15279 15281 15283 15284 15285 15286 15287 15288 
##     D     D     D     D     D     D     D     C     D     C     D     D 
## 15289 15290 15291 15295 15298 15299 15301 15302 15303 15304 15310 15311 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15316 15320 15322 15324 15326 15327 15333 15335 15338 15339 15345 15347 
##     D     D     C     C     C     C     D     D     D     D     D     D 
## 15348 15349 15350 15356 15360 15362 15364 15365 15366 15369 15377 15379 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15381 15388 15389 15391 15403 15404 15405 15414 15433 15434 15436 15437 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15448 15449 15454 15456 15458 15463 15471 15473 15474 15481 15488 15492 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15497 15498 15500 15504 15512 15514 15516 15519 15520 15521 15523 15529 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15530 15531 15532 15537 15539 15542 15550 15552 15555 15565 15566 15571 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15574 15575 15577 15578 15581 15584 15586 15592 15594 15598 15600 15601 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15606 15607 15608 15611 15612 15615 15619 15623 15629 15631 15634 15636 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15638 15641 15646 15648 15649 15650 15653 15656 15661 15664 15666 15668 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15678 15681 15686 15688 15690 15694 15696 15700 15703 15704 15706 15709 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15710 15712 15715 15718 15719 15724 15731 15733 15734 15740 15746 15755 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15757 15763 15773 15779 15785 15791 15793 15800 15803 15807 15808 15811 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15813 15814 15817 15819 15820 15823 15827 15833 15834 15837 15840 15841 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15842 15843 15852 15855 15860 15861 15862 15865 15879 15885 15886 15888 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15892 15893 15895 15897 15898 15899 15909 15915 15916 15924 15927 15930 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 15932 15942 15944 15945 15946 15957 15959 15961 15964 15971 15981 16002 
##     D     D     D     D     D     D     D     D     D     D     D     D 
## 16004 16006 16010 16016 16017 16028 16029 16031 16037 16038 16041 16045 
##     D     D     D     E     E     E     E     E     E     E     E     E 
## 16047 16048 16049 16058 16059 16060 16061 16064 16065 16068 16071 16072 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16076 16080 16087 16090 16091 16095 16098 16100 16101 16102 16108 16110 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16111 16114 16117 16120 16122 16123 16124 16127 16130 16135 16140 16142 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16144 16146 16147 16150 16151 16153 16156 16159 16162 16166 16168 16169 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16170 16172 16174 16179 16181 16182 16184 16187 16191 16202 16204 16207 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16210 16215 16219 16222 16227 16236 16237 16238 16241 16247 16250 16257 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16259 16261 16265 16266 16272 16277 16283 16286 16289 16292 16293 16294 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16296 16300 16303 16306 16309 16315 16320 16325 16331 16335 16336 16338 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16341 16342 16344 16348 16349 16350 16352 16363 16364 16365 16368 16373 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16374 16376 16380 16385 16386 16391 16392 16393 16399 16401 16403 16405 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16409 16410 16411 16420 16424 16429 16435 16439 16442 16444 16446 16447 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16451 16452 16453 16461 16467 16468 16472 16476 16479 16481 16486 16488 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16492 16493 16495 16498 16499 16503 16505 16507 16508 16512 16514 16515 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16518 16519 16521 16527 16530 16531 16533 16539 16543 16545 16546 16548 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16549 16551 16553 16559 16561 16570 16572 16573 16575 16576 16583 16587 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16592 16604 16611 16612 16614 16617 16619 16620 16621 16623 16624 16626 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16627 16628 16635 16637 16638 16640 16641 16645 16656 16664 16669 16672 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16673 16674 16675 16678 16679 16684 16689 16690 16692 16698 16702 16714 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16717 16718 16719 16720 16724 16726 16732 16736 16737 16741 16748 16757 
##     E     E     D     D     E     E     E     E     E     E     E     E 
## 16759 16769 16772 16774 16776 16789 16795 16796 16797 16799 16804 16806 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16811 16815 16817 16824 16825 16826 16828 16831 16834 16836 16837 16843 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16844 16847 16849 16851 16852 16861 16863 16864 16866 16868 16870 16877 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16880 16881 16882 16883 16891 16893 16897 16902 16904 16907 16909 16911 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16917 16918 16921 16922 16928 16930 16940 16942 16943 16944 16948 16950 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16952 16954 16955 16958 16962 16965 16966 16968 16969 16970 16980 16993 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 16995 17001 17007 17010 17013 17015 17022 17023 17029 17030 17031 17033 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17041 17042 17044 17045 17046 17048 17050 17051 17052 17057 17058 17060 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17061 17063 17069 17077 17078 17087 17095 17099 17101 17103 17104 17109 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17112 17113 17116 17122 17123 17125 17127 17128 17137 17140 17145 17147 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17150 17152 17155 17160 17163 17166 17177 17187 17188 17190 17191 17195 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17196 17202 17203 17204 17207 17208 17211 17218 17219 17228 17231 17235 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17236 17243 17246 17247 17248 17249 17252 17256 17259 17261 17262 17264 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17267 17271 17273 17275 17289 17294 17296 17297 17298 17300 17307 17308 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17311 17312 17320 17327 17331 17333 17335 17339 17340 17341 17352 17353 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17356 17359 17362 17364 17366 17367 17368 17371 17375 17378 17380 17381 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17388 17392 17396 17399 17403 17404 17405 17407 17410 17412 17415 17422 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17434 17440 17442 17443 17446 17451 17460 17467 17470 17474 17476 17486 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17488 17490 17494 17498 17505 17508 17514 17515 17518 17521 17522 17523 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17527 17530 17532 17535 17545 17547 17548 17550 17553 17557 17559 17567 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17570 17573 17575 17576 17582 17591 17595 17600 17601 17606 17612 17615 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17618 17620 17621 17622 17623 17628 17629 17633 17634 17637 17644 17659 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17664 17666 17669 17671 17672 17675 17676 17683 17684 17686 17687 17693 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17699 17705 17706 17707 17708 17709 17718 17720 17721 17728 17730 17731 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17733 17741 17749 17750 17751 17752 17753 17754 17755 17759 17761 17764 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17765 17769 17775 17782 17783 17785 17786 17789 17793 17795 17799 17802 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17804 17807 17809 17811 17812 17817 17820 17821 17830 17833 17834 17841 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17846 17848 17857 17865 17873 17878 17882 17886 17889 17890 17893 17898 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17899 17900 17901 17903 17904 17910 17911 17913 17918 17920 17923 17924 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17925 17927 17929 17933 17934 17939 17941 17943 17947 17948 17950 17952 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17954 17955 17960 17963 17965 17967 17977 17982 17984 17985 17988 17990 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 17992 17996 17998 18002 18003 18006 18008 18010 18016 18018 18019 18020 
##     E     D     D     E     E     E     E     E     E     E     E     E 
## 18027 18028 18032 18033 18035 18038 18040 18041 18043 18047 18049 18057 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18059 18064 18069 18070 18072 18080 18083 18084 18085 18091 18093 18094 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18103 18108 18109 18115 18117 18120 18122 18125 18127 18132 18133 18135 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18138 18140 18141 18151 18153 18157 18158 18162 18163 18171 18172 18174 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18176 18179 18181 18194 18195 18197 18201 18205 18206 18207 18209 18210 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18212 18216 18220 18223 18225 18234 18237 18239 18242 18252 18253 18271 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18280 18287 18291 18292 18294 18295 18296 18298 18299 18300 18304 18306 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18308 18315 18317 18318 18319 18320 18322 18323 18325 18329 18334 18335 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18336 18344 18349 18354 18356 18358 18364 18365 18369 18372 18377 18386 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18387 18396 18398 18404 18407 18408 18414 18415 18423 18425 18430 18434 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18436 18440 18442 18445 18446 18447 18455 18457 18458 18460 18461 18465 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18476 18478 18479 18482 18486 18489 18491 18494 18497 18502 18510 18511 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18518 18521 18524 18531 18532 18534 18540 18541 18543 18552 18556 18561 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18562 18567 18571 18575 18576 18578 18579 18582 18585 18586 18591 18597 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18598 18599 18601 18606 18608 18615 18616 18617 18621 18626 18629 18630 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18632 18633 18634 18637 18642 18646 18655 18658 18661 18662 18668 18669 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18670 18671 18675 18676 18678 18683 18691 18694 18701 18702 18707 18711 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18712 18716 18717 18724 18730 18733 18734 18739 18740 18743 18749 18755 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18758 18760 18764 18765 18770 18775 18779 18781 18790 18791 18795 18802 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18803 18808 18812 18817 18820 18822 18825 18829 18831 18838 18839 18841 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18843 18846 18847 18854 18855 18856 18857 18865 18868 18873 18878 18881 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18882 18885 18886 18887 18892 18893 18898 18899 18902 18908 18910 18914 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18916 18917 18923 18926 18927 18928 18933 18934 18935 18938 18943 18949 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18950 18951 18958 18959 18960 18962 18965 18973 18975 18979 18981 18987 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 18988 18993 18999 19000 19001 19003 19006 19010 19011 19014 19015 19016 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19018 19025 19033 19035 19041 19061 19064 19066 19070 19075 19078 19082 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19085 19087 19091 19099 19101 19104 19105 19107 19108 19110 19111 19113 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19122 19125 19126 19127 19128 19132 19138 19139 19141 19157 19158 19161 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19163 19168 19169 19172 19176 19178 19179 19180 19183 19186 19187 19192 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19193 19195 19210 19211 19214 19217 19218 19219 19222 19229 19233 19235 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19237 19240 19243 19245 19246 19247 19253 19254 19256 19257 19263 19266 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19273 19277 19278 19281 19289 19291 19295 19298 19300 19301 19303 19304 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19311 19313 19324 19327 19330 19336 19343 19350 19353 19355 19359 19372 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19380 19381 19384 19385 19392 19399 19406 19407 19413 19420 19422 19425 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19430 19438 19440 19441 19442 19445 19447 19451 19455 19456 19458 19462 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19465 19472 19473 19477 19478 19481 19482 19484 19489 19493 19495 19503 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19504 19506 19514 19515 19518 19520 19522 19530 19537 19543 19545 19549 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19550 19558 19559 19561 19563 19567 19568 19575 19578 19588 19595 19603 
##     E     E     E     E     E     E     E     E     E     E     E     E 
## 19607 19613 19617 19618 19621 
##     E     E     E     E     E 
## Levels: A B C D E
```
