#-------------------------------------------------------------------------
# AUTHOR: David Malone
# FILENAME: decision_tree_2.py
# SPECIFICATION: Testing how data set size impacts decision tree accuracy
# FOR: CS 4210- Assignment #2
# TIME SPENT: 4 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn import tree
import pandas as pd

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

#Reading the test data in a csv file using pandas
dbTest = []
df_test = pd.read_csv('contact_lens_test.csv')
for _, row in df_test.iterrows():
    dbTest.append(row.tolist())

def map_age(x):
    m = {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3}
    return m[x]

def map_specs(x):
    m = {'Myope': 1, 'Hypermetrope': 2}
    return m[x]

def map_astig(x):
    m = {'Yes': 1, 'No': 2}
    return m[x]

def map_tpr(x):
    m = {'Normal': 1, 'Reduced': 2}
    return m[x]

def map_all(x):
    age = x[0]
    specs = x[1]
    astig = x[2]
    tpr = x[3]
    return [map_age(age), map_specs(specs), map_astig(astig), map_tpr(tpr)]

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []
    accuracy = 0

    #Reading the training data in a csv file using pandas
    # --> add your Python code here
    df_train = pd.read_csv(ds)
    for _, row in df_train.iterrows():
        dbTraining.append(row.tolist())

    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
    x_raw = [row[:4] for row in dbTraining]
    
    X = list(map(map_all, x_raw))


    #Transform the original categorical training classes to numbers and add to the vector Y.
    #For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    def concat_list(l):
        concat = []
        for i in l:
            concat += i
        return concat
    
    def map_y(y):
        m = {'Yes': 1, 'No': 2}
        return [m[i] for i in y]
    
    y_raw = concat_list([row[4:] for row in dbTraining])
    Y = map_y(y_raw)

    accuracy_total = 0
    
    #Loop your training and test tasks 10 times here
    for i in range (10):
        correct_guesses = 0

        # fitting the decision tree to the data using entropy as your impurity measure and maximum depth = 5
        # --> addd your Python code here
        clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5).fit(X,Y)
        
        #Read the test data and add this data to dbTest
        #--> add your Python code here

        for data in dbTest:
            #Transform the features of the test instances to numbers following the same strategy done during training,
            #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
            X_test = [map_age(data[0]), map_specs(data[1]), map_astig(data[2]), map_tpr(data[3])]
            y_pred = clf.predict([X_test])[0]
            #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            #--> add your Python code here
        
            #Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            #--> add your Python code here
            m = {'Yes': 1, 'No': 2}
            y_true = m[data[4]]
            if y_pred == y_true:
                correct_guesses += 1

        accuracy_total += correct_guesses / len(dbTest)

    #Find the average of this model during the 10 runs (training and test set)
    #--> add your Python code here
    avg_accuracy = accuracy_total / 10

    #Print the average accuracy of this model during the 10 runs (training and test set).
    #Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    print(f"final accuracy when training on {ds}: {avg_accuracy}")




