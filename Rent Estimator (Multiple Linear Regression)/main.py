import pandas as pd
import sklearn
from sklearn import preprocessing, linear_model
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv("houses_to_rent.csv")

### LOADING THE DATA ###
print('-'*30);print("IMPORTING DATA");print('-'*30)
data = data[['city','rooms','bathroom','parking spaces','fire insurance','furniture','rent amount']]
# print(data.head(5))

### CLEANING THE DATA ###
'''Clearly the CITY column is changed to numbers i.e. 1 for in city and 0 for not in city.
 We need to do the same for the FLOOR column.
Since ML doesn't understand strings.'''

data['rent amount'] = data['rent amount'].map(lambda i:int(i[2:].replace(',','')))   ##So this will remove R$ symbol and "," char from the rent amount and convert it to integer from string
data['fire insurance'] = data['fire insurance'].map(lambda i:int(i[2:].replace(',','')))

le = preprocessing.LabelEncoder()
data['furniture'] = le.fit_transform(data['furniture'])
print(data.head(5))

print('-'*30);print("CHECKING NULL DATA");print('-'*30)
print(data.isnull().sum())

### CLEANING THE DATA ###
print('-'*30);print("SPLITING NULL DATA");print('-'*30)
x = np.array(data.drop(['rent amount'],1))
y = np.array(data["rent amount"])
print(f"X: {x.shape}")
print(f"Y: {y.shape}")


xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(x,y,test_size = 0.2,random_state=10)

print('xTrain:',xTrain.shape)
print('yTest:',xTest.shape)

### TRAINING THE DATA ###
print('-'*30);print("TRAINING");print('-'*30)
model = linear_model.LinearRegression()
model.fit(xTrain,yTrain)
accuracy = model.score(xTest,yTest)
print(f"Coefficients: {model.coef_}")
print(f"Intercepts: {model.intercept_}")
print(f"Accuracy: {round(accuracy*100,3)}%")


### TESTING THE DATA ###
print('-'*30);print("MANUAL TESTING");print('-'*30)
testVals = model.predict(xTest)
print(testVals.shape)
error = []
# for i,testval in enumerate(testVals):
#     error.append(yTest[i]-testval)
#     print(f"Actual:{yTest[i]} Prediction:{int(testval)} Error: {int(error[i])}")

plt.title("Error Visualization\n(Tough to Visualize as it depends on 6 variable)")
plt.plot(xTest,yTest)
plt.plot(xTest,testVals)
# plt.legend()
plt.show()

