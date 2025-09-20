#-------------------------------------------------------------------------
# AUTHOR: David Malone
# FILENAME: decision_tree.py
# SPECIFICATION: Takes in a .csv file with data measuring age, glasses prescription, astigmatism, and tear production; outputs a decision tree recommending contact lenses
# FOR: CS 4210- Assignment #1
# TIME SPENT: 6 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append(row)

#encode the original categorical training features into numbers and add to the 4D array X.
#--> add your Python code here
x_raw = [row[:4] for row in db]
print(x_raw)

def map_age(x):
    m = {'Young': 2, 'Prepresbyopic': 1, 'Presbyopic': 0}
    return m[x]

def map_specs(x):
    m = {'Myope': 1, 'Hypermetrope': 0}
    return m[x]

def map_astig(x):
    m = {'No': 1, 'Yes': 0}
    return m[x]

def map_tpr(x):
    m = {'Normal': 1, 'Reduced': 0}
    return m[x]

def map_all(x):
    age = x[0]
    specs = x[1]
    astig = x[2]
    tpr = x[3]
    return [map_age(age), map_specs(specs), map_astig(astig), map_tpr(tpr)]

X = list(map(map_all, x_raw))

#encode the original categorical training classes into numbers and add to the vector Y.
#--> addd your Python code here
def concat_list(l):
    concat = []
    for i in l:
        concat += i
    return concat

def map_y(y):
    m = {'Yes': 0, 'No': 1} # the order of class names when plotting the tree
    return [m[i] for i in y]

y_raw = concat_list([row[4:] for row in db])
Y = map_y(y_raw)

#fitting the decision tree to the data using entropy as your impurity measure
#--> add your Python code here
clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(X, Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()