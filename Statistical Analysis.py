import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sn

#Get current directory. Read the CSV dataset as a Pandas Dataframe
#This works if the dataset is a CSV file located in the same folder as the python file


Current_cd = os.getcwd()
F_list = os.listdir(Current_cd)
dataset = glob.glob(os.path.join(Current_cd, "*.csv"))
dst = pd.read_csv(dataset[0])


#Get the dataframne schema
#print(dst.info())
#col1="Company"
#print(dst.columns[0])
#print(dst.iloc[:,0])

print(dst.describe())



#Save a PDF file with an histogram for each column of the dataset
def distribution(dataset):
    pdfFile = PdfPages('Histograms.pdf')

    n_plots = len(dataset.columns)
    for i in range(n_plots):
        fig,ax = plt.subplots(figsize=(12,12))
        ax.hist(dataset.iloc[:,i])
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_xlabel("{}".format(dataset.columns[i]), fontsize=12)
        ax.set_title("{} Histogram".format(dataset.columns[i]))
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right', fontsize=8)
        pdfFile.savefig(fig)
        plt.close()

    #plt.show()
    pdfFile.close()

distribution(dst)


##Transform categorical values into numerical values to get a correlation matrix
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
        dst = dst.replace(temp_col_dict).infer_objects(copy=False)
        #print(dst[dst.columns[i]].unique())

    #In case the data type of the first element of the current column is not a str, then do nothing    
    else:
        None

#Save a heatmap as a jpg in the same folder
fig,ax = plt.subplots(figsize=(24,16))
ax = sn.heatmap(dst.corr(), cmap="YlGnBu", annot=True)
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right', fontsize=8)
plt.savefig('Heatmap.jpg')
plt.close()


