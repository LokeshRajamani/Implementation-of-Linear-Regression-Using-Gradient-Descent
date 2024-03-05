# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Initialize weights randomly.

2.Compute predicted values.

3.Compute gradient of loss function.

4.Update weights using gradient descent.

## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: LOKESH R
RegisterNumber: 212222240055
```

```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
X=np.c_[np.ones(len(X1)),X1]
theta=np.zeros(X.shape[1]).reshape(-1,1)
    
for _ in range(num_iters):
predictions=(X).dot(theta).reshape(-1,1)
errors=(predictions-y).reshape(-1,1)
theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
return theta

data=pd.read_csv("C:/Users/admin/Downloads/50_Startups.csv",header=None)
data.head()

X=(data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)

theta=linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
*
```
## Output:
![Untitled-3](https://github.com/LokeshRajamani/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/120544804/d2484f09-254f-4e84-97a9-10ece4815db1)


![Untitled-1](https://github.com/LokeshRajamani/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/120544804/a0f56c44-6263-43ea-a732-d8b279b1e95e)


![Untitled-2](https://github.com/LokeshRajamani/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/120544804/ab1eaa09-5ae6-45e0-a7e5-f92c3d5b64bf)


![Untitled](https://github.com/LokeshRajamani/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/120544804/d999433a-e63b-4ea4-9293-2a25bb577687)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
