# Implementation-of-Linear-Regression-Using-Gradient-Descent
# Date: 05.02.2024
## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize weights randomly.
2. Compute predicted values.
3. Compute gradient of loss function.
4. Update weights using gradient descent.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Lokesh R
RegisterNumber:  212222240055
*/
```

```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        
        #calculate predictions
        predictions = (X).dot(theta).reshape(-1,1)
        
        #calculate errors
        errors=(predictions - y ).reshape(-1,1)
        
        #update theta using gradiant descent
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
                                        
data=pd.read_csv("C:/classes/ML/50_Startups.csv")
data.head()

#assuming the lost column is your target variable 'y' 

X = (data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler = StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
print(X)
print(X1_Scaled)

#learn modwl paramerers

theta=linear_regression(X1_Scaled,Y1_Scaled)

#predict target value for a new data
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")

```

## Output:
### data information
![image](https://github.com/Ashwinkumar-03/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118663725/13bc974b-a702-46d8-875d-923f9e27129a)

### value of X:
![image](https://github.com/Ashwinkumar-03/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118663725/2d2fdcd5-406c-4f5e-90c0-920c4e9afa0f)

### value of X1_Scaled:
![image](https://github.com/Ashwinkumar-03/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118663725/4743c849-ac35-42d2-b944-de07bae9474e)

### predicted value:
![image](https://github.com/Ashwinkumar-03/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118663725/39fd799a-7a2c-4e79-bc3b-00528b4d4a9f)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
