# ML Model to predict Diabetes - Training, Testing, Auditing
## Introduction:
Diabetes is a widespread health concern, impacting individuals and societies worldwide. The significance of this problem lies in its potential complications, including cardiovascular diseases, kidney failure, and other serious health issues. In the context of healthcare decisions, ML plays a pivotal role in predicting whether an individual is at risk of developing diabetes. 
The decisions ML models have to make involve processing various data, such as glucose levels, age, and BMI, to classify individuals as either having or not having diabetes.
## Goal:
This study aims to develop, train, test, evaluate, and ethically audit a Random Forest model designed to predict diabetes. The primary goal is to provide a reliable and accurate tool that aids healthcare professionals in identifying individuals at risk. By incorporating ethical considerations and transparency in our model, we address concerns related to the responsible use of ML in healthcare. Ultimately, our study seeks to contribute to the advancement of diabetes prediction and management through the responsible application of machine learning techniques. 
## Data:
Pima Indian Diabetes dataset of the National Institute of Diabetes and Digestive and Kidney Diseases is used for this project. The dataset primarily contains diagnostic measurements of Pima Indian females 21 years and older. <br />
There are a total of 768 samples and 8 features : Glucose ; BMI ; Blood preassure ; Pregnancy ; Skin Thickness ; Insulin ; Pedigree Function ; Age
## Work Flow:
In order to come up with the most optimal parameters and hyper parameters for this RF model, the following ranges are used to perform analysis:<br /> 
Values chosen for analysis of optimal ntree (n_estimators) = [100, 500, 1000] <br /> 
Values chosen for analysis of optimal mtry (max_features) = [2, 3, 4, 6] <br /> 
Total #features = 8. So, sqrt(8) = 2.83. Hence, chose values around this number for ntree. <br />
the optimal values are: ntree = 1000  ;  mtry = 3  with OOB accuracy = 0.76 Avg. accuracy using 3 fold CV = 0.78   F1 score = 0.67 <br />
Using these ntree and mtry values, we trained and tested 3 models to find the best model: 
1. Using the original dataset as it is (in original dataset, the missing values were replaced with 0’s) 
2. Removing the features that had > 10% missing values (removed 2 features) 
3. Replacing all missing values with the median value of that feature.
## Training Results:
Comparing all three models to find the best model: <br/> ![image](https://github.com/hru952/DiabetesModel/assets/124914776/f6b94e87-d3aa-4b96-a8ff-a36337cb6771) <br/>
Based on the results, it is concluded that Model 1 would be the most optimal for using as run-time engine. <br/>
The confusion matrix and ROC curve are as follows: ![image](https://github.com/hru952/DiabetesModel/assets/124914776/fbaefcde-c6c1-44d1-b461-80992c4a7c38)
![image](https://github.com/hru952/DiabetesModel/assets/124914776/72f0d608-ca1c-412d-9c3c-460a6141e75f)
![image](https://github.com/hru952/DiabetesModel/assets/124914776/ec99fe1a-e2dd-4bd8-9b6f-dda2caf6fac6)
## Model Prediction and Confidence:
1 random positive class sample and one random negative class sample were picked. The trained runtime model predicted their output labels as follows: <br/> ![image](https://github.com/hru952/DiabetesModel/assets/124914776/774fcbf8-3b70-42d8-8d52-9659af9be513) <br/>
It can be seen that for the positive class label, 97.6% of the trees voted correctly and for the negative class sample, 91.4% of the trees voted correctly. For this application, it is very important to correctly predict the positive class more than the negative class. As shown, the probability of voting is very high for the positive class (almost 98%) as desired. <br/>
The model confidence is calculated as follows: average of probability for correctly classified positive class samples among all the correctly classified samples. The model confidence is obtained as 0.82 or 82%, which only justifies that the model is working well.




