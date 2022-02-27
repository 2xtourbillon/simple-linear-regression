# import the packages
from tkinter import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt

# reading the file and assigning X, y
def readFile(file_path):
    dataset = pd.read_csv(file_path)
    X = dataset.iloc[:,0].values # Land Area (sqft)
    y = dataset.iloc[:,1].values # Purchase Price ($M)
    return X, y, dataset

file_path = r'data/landprice.csv'
X, y, df = readFile(file_path)

# splitting the data
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_data(X, y)

# training the model
def model_train(X_train, y_train, X_test, y_test):
    
    X_train1 = np.reshape(X_train, (-1, 1))
    y_train1 = np.reshape(y_train, (-1, 1))

    X_test1 = np.reshape(X_test, (-1, 1))
    y_test1 = np.reshape(y_test, (-1, 1))
    
    lin_regressor = LinearRegression()
    lin_regressor.fit(X_train1, y_train1)

    y_pred = lin_regressor.predict(X_test1)

    return y_pred

y_pred = model_train(X_train, y_train, X_test, y_test)

# initialize tkinter window
window = Tk()
window.geometry('600x700')
window.title('Template Window')
label = Label(window, text="Enter the Area of the Land in '000 sqft", fg='red', font='Courier 15')
label.pack()

# Area entry field
area = StringVar()
area.set('')
entry = Entry(window, textvariable=area, fg='green', width=10, font='Courier 15')
entry.pack()

# Prediction button
pred_button = Button(window, text='Predict', fg='red', command=model_pred, height=2, width=15)
pred_button.pack()



mainloop()


