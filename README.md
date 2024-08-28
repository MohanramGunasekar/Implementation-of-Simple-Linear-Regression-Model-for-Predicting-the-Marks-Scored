# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Mohanram Gunasekar
RegisterNumber:212223240095
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
![362105691-16b87f3e-43cb-41da-b42d-6baf6ad64fa1](https://github.com/user-attachments/assets/81a28b69-394b-4579-8f7f-7015bbaec315)

![362106506-5210ac75-2294-4c1a-8300-d60c5fc15495](https://github.com/user-attachments/assets/25230556-ec84-4fbe-a176-9cab2cec39ec)

![362106564-88718607-0fb4-4670-9d78-1853a963eced](https://github.com/user-attachments/assets/89990b8e-733a-4a0b-bef7-830ceeaf8889)

![362105733-0acd20fa-03b0-4cbf-8269-e8e35a798ea1](https://github.com/user-attachments/assets/464b4c74-c52d-472c-b89c-0fee96480bf3)

![362105755-7f43b928-8a38-4f79-9425-1a22fc530353](https://github.com/user-attachments/assets/df433190-b279-4d26-b551-a4acec3cb175)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
