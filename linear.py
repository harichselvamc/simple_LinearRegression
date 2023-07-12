# import the nessary libraries
import numpy
import matplotlib.pyplot as plot
import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# import the dataset
dataset=pandas.read_csv('salaryData.csv')
print(dataset)


x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)



print(dataset.head())


print(dataset.shape)
print(x.shape)
print(y.shape)



print(type(dataset))
print(type(x))
print(type(y))


cc=dataset.corr()
print(cc)

#pip install seaborn

import seaborn as sns
#sns heatmap(data,square=True,annot=True)
sns.heatmap(cc,vmax=1,square=True,annot=True,cmap="coolwarm")


# splitting the dataset into the training set and test set
# 
# we're splitting the data in 1/3 , so out of 30 rows,20rows will go into the training set and 10 rows will go to the testing set.



xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=1/3,random_state=0)

# creating a LinearRegression object and fitting it with our training set

lr=LinearRegression()
lr.fit(xtrain,ytrain)

print(lr)

print(xtest)
print(ytest)


# predict the test set results

yprediction=lr.predict(xtest) #y test y predictions

print(yprediction)
print(ytest)



lr.coef_
print(lr.coef_)
lr.intercept_
print(lr.intercept_)


# predict

pred_salary=lr.predict([[6]])
print(pred_salary)


# Visualising the Training set results
plot.scatter(xtrain,ytrain,color='red')
plot.plot(xtrain,lr.predict(xtrain),color='blue')
plot.title("salary vs experience (training set)")
plot.xlabel("Years of Experience ")
plot.ylabel("salary")
plot.show()


# Visualising the test set results
plot.scatter(xtest,ytest,color='Red')
plot.plot(xtest,lr.predict(xtest),color='blue') 
plot.title("salary vs experience (test set)")
plot.xlabel("Years of Experience ")
plot.ylabel("salary")
plot.show()




lr.predict([[4.2]])

import pickle

# Save the model to a file
with open('salarypredict.pkl', 'wb') as file:
    pickle.dump("salarypredict.pkl", file)

import joblib
filename="salarypredict.pkl"

joblib.dump(lr,filename)
load_model=joblib.load(filename)
p=load_model.predict([[5.5]])
print(p)


