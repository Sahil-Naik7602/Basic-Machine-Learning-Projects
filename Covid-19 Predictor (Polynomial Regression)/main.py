import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

### CLEANING DATA ###
# print("-"*30);print("DATA LOADING");print("-"*30)
# data = pd.read_csv('covid-cases-India.csv')
# new_data = data[["Unnamed: 0","total_cases"]]

# new_data.rename(columns = {'Unnamed: 0':'id'}, inplace= True)
# new_data.to_csv("Authenticated_cleansed_data.csv")
# # print(new_data.head(100).tail(5))

### LOADING DATA ###
print("-"*30);print("DATA LOADING");print("-"*30)
data = pd.read_csv("Authenticated_cleansed_data.csv")
print(data.head(10))

### PREPARE THE DATA ###
print("-"*30);print("DATA PREPARING");print("-"*30)
x = np.array(data['id']).reshape(-1,1)
y = np.array(data['total_cases']).reshape(-1,1)
plt.plot(y,'-m',label="Original")
polyFeat = PolynomialFeatures(degree=7) ##7 gives the best prediction with maximum accuracy
x = polyFeat.fit_transform(x)


### TRAINING DATA ###
'''Since we don't have enough data for training and testing we are using all data for training'''
print("-"*30);print("DATA LOADING");print("-"*30)
model = linear_model.LinearRegression()
model.fit(x,y)
accuracy = model.score(x,y)
print(f"Accuracy: {round(accuracy*100,3)}%")
y0 = model.predict(x)
# plt.plot(y0,"--b",label = "Prediction")


### PREDICTION ###
print("-"*30);print("PREDICTION");print("-"*30)
days = int(input("Enter the number of days to know the total number of increases upto\n(The days are counted from the last updated day):"))

print(f'Prediction Cases after {days} days:',end='')
print(round(int(model.predict(polyFeat.fit_transform([[751+days]])))/1000000,2),'Million')

x1 = np.array(list(range(1,751+days))).reshape(-1,1)
y1 = model.predict(polyFeat.fit_transform(x1))
plt.plot(y1,'--r', label = "Prediction")
plt.plot(y0,"--b",label = "Best Regression Curve")
plt.legend()
plt.show()









