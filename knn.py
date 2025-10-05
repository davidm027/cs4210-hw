#-------------------------------------------------------------------------
# AUTHOR: David Malone
# FILENAME: knn.py
# SPECIFICATION: Create a k-nearest neighbors algorithm classifying emails as spam or not spam
# FOR: CS 4210- Assignment #2
# TIME SPENT: 4 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

#Reading the data in a csv file using pandas
db = []
df = pd.read_csv('email_classification.csv')
for _, row in df.iterrows():
    db.append(row.tolist())

errors = 0

#Loop your data to allow each instance to be your test set
for i in db:

    #Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    db_train = [row for row in db if row != i] # everything that isn't our current instance is our training set

    x_raw = [sample[:-1] for sample in db_train]
    X_train = pd.DataFrame(data=x_raw, dtype=np.float64)

    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    def concat_list(l):
        concat = []
        for i in l:
            concat += i
        return concat

    def map_y(y):
        m = {'ham': 0.0, 'spam': 1.0}
        return m[y]

    y_raw = concat_list([row[-1:] for row in db_train])
    Y_train = list(map(map_y, y_raw))

    #Store the test sample of this iteration in the vector testSample
    #--> add your Python code here

    #Fitting the knn to the data using k = 1 and Euclidean distance (L2 norm)
    #--> add your Python code here
    clf = KNeighborsClassifier(n_neighbors=1, metric='l2').fit(X_train, Y_train)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2, 3, 4, 5, ..., 20]])[0]
    #--> add your Python code here
    X_test = list(map(float, i[:-1]))
    Y_true = map_y(i[-1])

    class_predicted = clf.predict([X_test]).item()

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if class_predicted != Y_true:
        errors += 1

#Print the error rate
#--> add your Python code here
print(f"Error rate: {errors/100}")
