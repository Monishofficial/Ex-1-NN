## ENTER YOUR NAME: MONISH N
## ENTER YOUR REGISTER NO: 212223240097
## EX. NO: 1
## DATE: 07/03/2025
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:
To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:
**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**
Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**
For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1: Importing the libraries<BR>
STEP 2: Importing the dataset<BR>
STEP 3: Taking care of missing data<BR>
STEP 4: Encoding categorical data<BR>
STEP 5: Normalizing the data<BR>
STEP 6: Splitting the data into test and train<BR>

##  PROGRAM:
```
from google.colab import files
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('Churn_Modelling.csv')
print(df)

df.head()
df.tail()
df.columns

print(df.isnull().sum())
df.duplicated()

X = df.iloc[:, :-1].values
print(X)
y = df.iloc[:,-1].values
print(y)

df.fillna(df.mean().round(1), inplace=True)
print(df.isnull().sum())

df.describe()
df1 = df.drop(['Surname','Geography','Gender'],axis=1)
df1.head()

scaler = MinMaxScaler()
df2 = pd.DataFrame(scaler.fit_transform(df1))
print(df2)

X = df.iloc[:, :-1].values
print(X)
y = df.iloc[:,-1].values
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
print(X_train)
print("Length of X_train:",len(X_train))
print(X_test)
print("Length of X_test:",len(X_test))
```

## OUTPUT:
### DATA HEAD 
![image](https://github.com/Lavanyajoyce/Ex-1-NN/assets/118708024/b6421015-1968-410c-bea8-96e25b836dc3)
### DATA CHECKING
![image](https://github.com/Lavanyajoyce/Ex-1-NN/assets/118708024/bc61200c-c9c4-4069-ac66-5298d81c9af6)
### NULL VALUES
![image](https://github.com/Lavanyajoyce/Ex-1-NN/assets/118708024/31c7b09e-6cf8-4fd4-b50e-70c594c7f68a)
### X VALUE
![image](https://github.com/R-Udayakumar/Ex-1-NN/assets/118708024/52125358-7674-445f-ab3c-3c14b6ffdecc)
### Y VALUE
![image](https://github.com/R-Udayakumar/Ex-1-NN/assets/118708024/1b782e02-58f2-4c18-a36e-af887571ac76)
### OUTLIERS
![image](https://github.com/R-Udayakumar/Ex-1-NN/assets/118708024/875b36a6-7c2f-440e-b903-68a64c1e0d9f)
### DROP
![image](https://github.com/R-Udayakumar/Ex-1-NN/assets/118708024/5b33f2eb-0eb4-499c-a68e-a9de4dfeeb0a)
### NORMALIZATION
![image](https://github.com/R-Udayakumar/Ex-1-NN/assets/118708024/b86e4907-7082-46d5-ac42-d60ead60433e)
### DATA SPLITING
![image](https://github.com/R-Udayakumar/Ex-1-NN/assets/118708024/f8c2564a-9335-4e2a-af63-c47a856079f4)
### TRAINING & TEST DATA 
![image](https://github.com/R-Udayakumar/Ex-1-NN/assets/118708024/ec63aef9-fb98-4190-85a0-1aefc9b7fed6)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


