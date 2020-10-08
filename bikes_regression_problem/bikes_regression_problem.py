
############## Regression_problem # Predict the number of bikees
## Imported Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read Data as DataFrame
df = pd.read_csv('bikes.csv', sep=',', na_values=['N/A', 'no', '?'])
#Return the first n rows.
print(df.head(10))           # n= 10
#method for prints information about a DataFrame including the index dtype and columns, non-null values and memory usage
df.info()  #datatype of all columns float  Except date column
# Check Missing Data
print(df.isnull().sum()) # not contain missing data
print(df.describe())
# To Visualize Data
# sns.pairplot(df)
# plt.show()
## Date and Time features convert  date column from object to datetime
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors="coerce")   ##formate of date column is  year- month- day
df.info()
## Extract features from date column
df['Year'] = df['date'].dt.year
df['Month'] = df['date'].dt.month
df['Day'] =df['date'].dt.day_name()
## from month column extract season
def map_months(x):
    if x in [12, 1, 2]:
        return 'Winter'
    elif x in [3, 4, 5]:
        return 'Spring'
    elif x in [6, 7, 8]:
        return 'Summer'
    elif x in [9, 10, 11]:
        return 'Autumn'

df['Season'] = df['Month'].apply(map_months)
print(df.head(10))

### Visualization
sns.pairplot(df, vars=['temperature', 'humidity', 'windspeed', 'count', 'Season'])
plt.show()
################# Detect and Handle Outliers
columns =['temperature', 'humidity', 'windspeed']

from datasist.structdata import detect_outliers
outliers = detect_outliers(df, 0, columns)
print(len(outliers)) # 10 number of rows which contain outliers
# delet outliers
df.drop(outliers, inplace=True)
## Deal with Categorical Data (Season , month)
df = pd.get_dummies(df, columns=['Season'], drop_first=True)
print(df.head(10))
#############  Feature Scaling
# # #####To split Data to train && test when using all data set linear or non linear relation
# # ### when select features
# #
from sklearn.model_selection import train_test_split
print(df['Year'])
x = df.drop(['count','date','Month', 'Day','Season_Summer'], axis=1)
y = df['count']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=90)
print(x_test.columns)
####### Train model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(x_train, y_train)
print(model.score(x_test, y_test)) # accuracy = 74% when n_estimator= 1000 and used column year, accuracy = 75% when n_estimator= 4000

### to save model
import joblib
joblib.dump(model, 'model.pkl')

#to load pretrained model
model = joblib.load('model.pkl')
print(int(round(model.predict([[2.896673, 54.267219, 15.136882, 2011,1, 0]])[0])))