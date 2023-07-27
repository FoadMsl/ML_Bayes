#=================================================
#   ML_HW2-2_Moslem_401129902
#   Foad Moslem - PhD Student - Aerodynamics
#   Using Python 3.9.16
#=================================================

try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass


#%% Load the dataset
import pandas as pd
df = pd.read_csv("./diabetes_prediction_dataset.csv")


#%% Quick look at the data structure
df.head() # looking at the top 5 rows
df.info() # get a quick description of the data
print(df.isnull().sum()) # check missing data | there are no missing values in any of the columns since all values in the output are False.
print(df.duplicated().sum()) # check duplicated data | there are some duplicate rows in the dataset since some of the values are True.
df.describe() # shows a summary of the numerical attributes | some columns have outliers since the maximum value is significantly higher than the 75th percentile value. These columns are age, BMI, HbA1c_level, and blood_glucose_level.


#%% Data Preprocessing

# Plot before data preprocessing
import seaborn as sns
g = sns.pairplot(df, hue='diabetes', palette="Set2", height=2.5, corner=True)
g.map_diag(sns.kdeplot)
g.map_lower(sns.scatterplot)
g.add_legend(frameon=True)

# drop duplicates
df.drop_duplicates(inplace=True)
print(df.duplicated().any()) # check for duplicates again

# Categoty Analysis
    ## gender
df["gender"].value_counts() # find out what category exist
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder() # Label Encoding for gender column
df["gender"] = label_encoder.fit_transform(df["gender"])

    ## smoking_history
df["smoking_history"].value_counts() # find out what category exist
smoking_history_mapping = {'never': 0, 'No Info': -1, 'current': 2, 'former': 1, 
                           'ever': 2, 'not current': 0} # Convert smoking history to numerical format
df["smoking_history"] = df["smoking_history"].map(smoking_history_mapping)

    ## age 
df['age'].hist() # hist plot of age | min value for age is 0.08, that is not possible! Remove all records where age is given in decimal
df = df[df["age"].mod(1)==0]
df["age"] = df["age"].astype(int) # convert age column datatype to int


#%% Another Quick look at the data structure
df =df.reset_index(drop=True) # Reset index
df.head()
df.info()
df

import seaborn as sns

# line plot between age and bmi
sns.lineplot(df, x = 'age', y = 'bmi', hue = 'diabetes', palette="Set2")

# Check smoking_history and diabetes relationship
sns.countplot(df, x="smoking_history", hue="diabetes", palette="Set2")
    ## Drop smoking history
df1 = df.drop('smoking_history', axis=1).reset_index(drop=True) # Reset index after drop

# The Last Quick look at the data structure
""" The Last Quick look at the data structure """
df1.head()
df1.info()
df1


#%% Data vitalization and Preprocessing

# Visualize the relationship between variables using scatter plot
g = sns.pairplot(df1, hue='diabetes', palette="Set2", height=2.5, corner=True)
g.map_diag(sns.kdeplot)
g.map_lower(sns.scatterplot)
g.add_legend(frameon=True)

# Split dataset for Train and Test, 70% and 30% of data respectively
X = df1.iloc[:,:-1].values
y = df1.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

# feature scaling
from sklearn import preprocessing
stand = preprocessing.StandardScaler() # Standardize features by removing the mean and scaling to unit variance.
X_train = stand.fit_transform(X_train) # Fit to data, then transform it
X_test = stand.transform(X_test) # Perform standardization by centering and scaling


#%% Modeling
# Training the model using Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB() 
gnb.fit(X_train, y_train) # The 'fit' method trains the algorithm on the training data, after the model is initialized.

y_pred_train = gnb.predict(X_train) # Predicts an train output
y_pred_test = gnb.predict(X_test) # Predicts an test output


#%% Evaluation

# Evaluate the model by Train and Test dataset
print("Train accuracy:", gnb.score(X_train, y_train)) # Return the mean accuracy on the given test data and labels.
print("Test accuracy:", gnb.score(X_test, y_test)) # Return the mean accuracy on the given test data and labels.

# Evaluate the model based on 4-fold cross validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(gnb, df1.iloc[:, :-1], df1.iloc[:, -1], cv=4) # trains and tests a model over multiple folds of dataset.
print("Cross-validation scores:", scores)
""" This cross validation method gives us a better understanding of model 
performance over the whole dataset instead of just a single train/test split."""

# Plot two confusion matrix for the classifier by Train and Test dataset.
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
    ## Train dataset
cm_train = confusion_matrix(y_train, y_pred_train)
sns.heatmap(cm_train, annot=True, cmap='Set2')
plt.title('Confusion matrix for Train dataset')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
    ## Test dataset
cm_test = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm_test, annot=True, cmap='Set2')
plt.title('Confusion matrix for Test dataset')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


#=================================================
#=================================================

#%% Model efficiency improvement (Extra credit)

# Select the best four features according to the Data vitalization part then make a new dataset.
df1.info()
best_features = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
df2 = df1[best_features + ['diabetes']]
df2.info()

# Retrain the model with the new dataset
X_new = df2.iloc[:,:-1].values
y_new = df2.iloc[:,-1].values
X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(X_new, y_new, train_size=0.7, random_state=42)
gnb_new = GaussianNB()
gnb_new.fit(X_new_train, y_new_train)

# Evaluate the new model with the new dataset.
print("Train accuracy with new dataset:", gnb_new.score(X_new_train, y_new_train))
print("Test accuracy with new dataset:", gnb_new.score(X_new_test, y_new_test))


#%% Apply thresholding method (Extra credit)

def thresholding_predict(model, X, threshold):
    return (model.predict_proba(X)[:, 1] > threshold).astype(int)

threshold = 0.4
y_pred_thresholded = thresholding_predict(gnb_new, X_new_test, threshold)
cm_thresholded = confusion_matrix(y_new_test, y_pred_thresholded)
print("Confusion matrix for Test dataset with thresholding:\n", cm_thresholded)

sns.heatmap(cm_thresholded, annot=True, cmap='Set2')
plt.title('Confusion matrix for Test dataset with threshold = 0.4')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()