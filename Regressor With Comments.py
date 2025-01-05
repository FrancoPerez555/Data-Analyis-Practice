import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import tree
from sklearn.neural_network import MLPRegressor


#Get current directory. Read the CSV dataset as a Pandas Dataframe
#This works if the dataset is a CSV file located in the same folder as the python file


Current_cd = os.getcwd()
F_list = os.listdir(Current_cd)
dataset = glob.glob(os.path.join(Current_cd, "*.csv"))
dst = pd.read_csv(dataset[0])
backup_dst = dst


#Explore the dataset and transform categorical data in
#numerical data so it can be fed to a regressor

#Iterate through the columns of the dataset
for i in range(len(dst.columns)):
    #Create empty dictionaries that will help transforming cathegorical values into numerical values
    temp_val_dict = {}
    temp_col_dict = {}

    #Get the type of the first element in the current column
    a = type(dst[dst.columns[i]][0])
    #print(a)
    #If the type is str, then the values are changed for numerical values
    if a == str:
        #print("Needs to change")
        #print(dst[dst.columns[i]].unique())
        #Get the unique cathegorical values of the column in a list
        key_list=dst[dst.columns[i]].unique()
        #Define a counter that will help replacing the cathegorical values
        x = 1

        for key in key_list:
            #Polulate temp_val_dict directory with the cathegorical values and associate those values with the current number of the counter
            temp_val_dict[key] = x
            x = x+1
        #print(temp_val_dict)
        #Define the key of temp_col_dict dictionary as the name of the current column and its value as the temp_val_dict dictionary
        temp_col_dict[str(dst.columns[i])]=temp_val_dict
        #print(temp_col_dict)
        #Replace the cathegorical values of the current columns with the ones assigned in the temp_val_dict dictionary using the replace() method
        dst = dst.replace(temp_col_dict)
        #print(dst[dst.columns[i]].unique())

    #In case the data type of the first element of the current column is not a str, then do nothing    
    else:
        None



#Convert the price column in a Values array
#Convert the rest of the columns in Features array
Y = dst['Price_euros'].to_numpy()
X = dst.drop("Price_euros", axis=1).to_numpy()

#Normalization of each column in the features array
for i in range(len(X[0,:])):
    num=(X[:,i]-(X[:,i].mean()))
    den= (X[:,i].std())
    norm = num/den
    X[:,i] = norm

#Splitting of the dataset into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3)

#Prediction using support vector machine
Model = svm.SVR()
Model.fit(X_train, Y_train)
Predictions = Model.predict(X_test)

#Prediction using decission tree regressor
Model2 = tree.DecisionTreeRegressor()
Model2.fit(X_train, Y_train)
Predictions_2 = Model2.predict(X_test)

#Prediction using a multilayered preceptron regressor
Model3 = MLPRegressor(activation='relu', learning_rate= 'adaptive', learning_rate_init= 0.0006, max_iter=1500)
Model3.fit(X_train, Y_train)
Predictions_3 = Model3.predict(X_test)


#The first 50 predictions of all regressors are plotted against the real values
plt.xlabel("Computer (Index)")
plt.ylabel("Prices")
plt.title("Some excersice")


plt.plot(Y_test[0:20], 'b', Predictions_2[0:20], 'g', Predictions[0:20], 'r', Predictions_3[0:20], 'c')
plt.legend(["Real values", "Decission Tree Predictions", "SVR Predictions", "Neural Network Regressor"], loc="lower right")
plt.show()

