import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import tree
from sklearn.neural_network import MLPRegressor

#Read the laptop prices csv as a pandas dataframe
dst = pd.read_csv('laptop_prices.csv')

#Explore the dataset and transform categorical data into numerical data so it can be fed to a regressor. This could also be done with scikit learn label encoder
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


#Definition of a funtion to fit the data to different models and get the prediction metrics
def predictor(mod, X_tr, Y_tr, X_ts, Y_ts):
    model = mod()
    model.fit(X_tr, Y_tr)
    results = model.predict(X_ts)
    model_mse = mean_squared_error(Y_ts, results)
    model_rmse = np.sqrt(model_mse)
    model_r2 = r2_score(Y_ts, results)

    print("{} model used for regression \nThe MSE is:  {}\nThe RMSE is: {}\nThe R2 Score is: {}\n".format(mod, model_mse, model_rmse, model_r2))

    return results, model_mse, model_rmse, model_r2

#In this case, SVR, DecisionTreeRegressor, MLP Repressor and Linear Regression are used without defining hyperparameters (for simplicity)
svr_pred, svr_mse, svr_rmse, svr_r2 = predictor(SVR, X_train, Y_train, X_test, Y_test)
tree_pred, tree_mse, tree_rmse, tree_r2 = predictor(DecisionTreeRegressor, X_train, Y_train, X_test, Y_test)
mlp_pred, mlp_mse, mlp_rmse, mlp_r2 = predictor(MLPRegressor, X_train, Y_train, X_test, Y_test)
lin_pred, lin_mse, lin_rmse, lin_r2 = predictor(LinearRegression, X_train, Y_train, X_test, Y_test)

#The first 25 predictions of each model are plotted against the real values
plt.plot(Y_test[0:20], 'b', svr_pred[0:20], 'g', tree_pred[0:20], 'r', mlp_pred[0:20], 'c',lin_pred[0:20], 'y')
plt.title("Predictions VS Real Values")
plt.ylabel("Price (Euros)")
plt.xlabel("Test instance")
plt.legend(["Real values", "SVR Predictions", "Decission Tree Predictions","MLP Predictions", "Linear Regression Prediction"], loc="lower right")
plt.show()
