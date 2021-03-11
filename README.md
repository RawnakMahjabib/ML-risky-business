# Machine Learning - Risky Business
 
![Credit Risk](Images/credit-risk.jpg)

## Background

Mortgages, student and auto loans, and debt consolidation are just a few examples of credit and loans that people seek online. Peer-to-peer lending services such as Loans Canada and Mogo let investors loan people money without using a bank. However, because investors always want to mitigate risk, a client has asked to help them predict credit risk with machine learning techniques.

In this assignment I built and evaluated several machine learning models to predict credit risk using data you'd typically see from peer-to-peer lending services. Credit risk is an inherently imbalanced classification problem (the number of good loans is much larger than the number of at-risk loans), so I employed different techniques for training and evaluating models with imbalanced classes. I used the imbalanced-learn and Scikit-learn libraries to build and evaluate models using the two following techniques:

1. [Resampling](#Resampling)
2. [Ensemble Learning](#Ensemble-Learning)

#### Resampling

In this Jupyter Notebook, I used the [imbalanced learn](https://imbalanced-learn.readthedocs.io) library to resample the LendingClub data, build and evaluate logistic regression classifiers using the resampled data.

The notebook consists of:

1. Reading the CSV into a DataFrame.

2. Splitting the data into Training and Testing sets.

3. Scaling the training and testing data using the `StandardScaler` from `sklearn.preprocessing`.

4. Using the provided code to run a Simple Logistic Regression:
    * Fitting the `logistic regression classifier`.
    * Calculating the `balanced accuracy score`.
    * Displaying the `confusion matrix`.
    * Printing the `imbalanced classification report`.

It also includes:

1. Oversampling the data using the `Naive Random Oversampler` and `SMOTE` algorithms.

2. Undersampling the data using the `Cluster Centroids` algorithm.

3. Over- and undersampling using a combination `SMOTEENN` algorithm.


For each of the above, I've:

1. Trained a `logistic regression classifier` from `sklearn.linear_model` using the resampled data.

2. Calculated the `balanced accuracy score` from `sklearn.metrics`.

3. Displayed the `confusion matrix` from `sklearn.metrics`.

4. Printed the `imbalanced classification report` from `imblearn.metrics`.


#### Ensemble Learning

In this section, I train and compared two different ensemble classifiers to predict loan risk and evaluate each model. I used the [Balanced Random Forest Classifier](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html#imblearn.ensemble.BalancedRandomForestClassifier) and the [Easy Ensemble Classifier](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.EasyEnsembleClassifier.html).

I begin by:

1. Reading the data into a DataFrame.

2. Splitting the data into training and testing sets.

3. Scaling the training and testing data using the `StandardScaler` from `sklearn.preprocessing`.


Then, I completed the following steps for each model:

1. Training the model using the quarterly data from LendingClub provided in the `Resource` folder.

2. Calculating the balanced accuracy score from `sklearn.metrics`.

3. Displaying the confusion matrix from `sklearn.metrics`.

4. Generating a classification report using the `imbalanced_classification_report` from imbalanced learn.

5. For the balanced random forest classifier only, I printed the feature importance sorted in descending order (most important feature to least important) along with the feature score.



