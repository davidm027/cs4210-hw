#-------------------------------------------------------------------------
# AUTHOR: David Malone
# FILENAME: naive_bayes.py
# SPECIFICATION: Create an Naive Bayes algorithm to evaluate tennis playing conditions
# FOR: CS 4210- Assignment #2
# TIME SPENT: 4 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd

dbTraining = []
dbTest = []

#Reading the training data using Pandas
df = pd.read_csv('weather_training.csv')
for _, row in df.iterrows():
    dbTraining.append(row.tolist())

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
x_train_raw = [row[1:-1] for row in dbTraining]

def map_outlook(x):
    m = {'Sunny': 1, 'Overcast': 2, 'Rain': 3}
    return m[x]

def map_temp(x):
    m = {'Cool': 1, 'Mild': 2, 'Hot': 3}
    return m[x]

def map_humidity(x):
    m = {'Normal': 1, 'High': 2}
    return m[x]

def map_wind(x):
    m = {'Weak': 1, 'Strong': 2}
    return m[x]

def map_all(x):
    outlook = x[0]
    temp = x[1]
    humidity = x[2]
    wind = x[3]
    return [map_outlook(outlook), map_temp(temp), map_humidity(humidity), map_wind(wind)]

X_train = list(map(map_all, x_train_raw))

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
def concat_list(l):
    concat = []
    for i in l:
        concat += i
    return concat

def map_y(y):
    m = {'Yes': 0, 'No': 1}
    return [m[i] for i in y]

y_train_raw = concat_list([row[-1:] for row in dbTraining])
Y_train = map_y(y_train_raw)

#Fitting the naive bayes to the data using smoothing
#--> add your Python code here
clf = GaussianNB()
clf.fit(X_train, Y_train)

#Reading the test data using Pandas
df = pd.read_csv('weather_test.csv')
for _, row in df.iterrows():
    dbTest.append(row.tolist())

#Printing the header os the solution
#--> add your Python code here
print(f"Day    Outlook   Temperature  Humidity  Wind    PlayTennis  Confidence")

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
x_test_raw = [row[:-1] for row in dbTest] # for displaying output only
x_test_raw_no_day = [row[1:-1] for row in dbTest]
X_test = list(map(map_all, x_test_raw_no_day))
Y_pred = clf.predict(X_test).tolist()
Y_probs = list(map(max, clf.predict_proba(X_test).tolist()))

results = list(zip(x_test_raw, Y_pred, Y_probs))

def print_results(rs):
    for r in rs:
        instance = r[0]
        day = instance[0]
        outlook = instance[1]
        temp = instance[2]
        humidity = instance[3]
        wind = instance[4]

        i_class = r[1]
        confidence = r[2]

        if confidence >= 0.75:
            print(f"{day:5}  {outlook:8}  {temp:11}  {humidity:8}  {wind:6}  {i_class:<11} {confidence:<10.2%}")

print_results(results)

