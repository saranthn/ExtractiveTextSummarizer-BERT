import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def linear_regression():
    # Importing the dataset
    dataset = pd.read_csv('data2.csv')
    x = dataset.iloc[:, 0].values.reshape(-1,1)
    y = dataset.iloc[:, 2].values
    print(x)
    print(y)

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

    # Fitting Simple Linear Regression to the Training set
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    # Visualizing the Training set results
    viz_train = plt
    viz_train.scatter(x_train, y_train, color='red')
    viz_train.plot(x_train, regressor.predict(x_train), color='blue')
    viz_train.title('bcss vs length of summary (Training set)')
    viz_train.xlabel('bcss')
    viz_train.ylabel('length of summary')
    viz_train.show()

    # Visualizing the Test set results
    viz_test = plt
    viz_test.scatter(x_test, y_test, color='red')
    viz_test.plot(x_train, regressor.predict(x_train), color='blue')
    viz_test.title('bcss vs length of summary (Test set)')
    viz_test.xlabel('bcss')
    viz_test.ylabel('length of summary')
    viz_test.show()

    y_pred = regressor.predict([[1]])
    print(y_pred)