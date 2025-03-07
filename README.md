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
```python
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
![Screenshot 2025-03-07 113129](https://github.com/user-attachments/assets/64ff6661-36d6-4691-9c0a-3a969b26313b)
### DATA TAIL
![Screenshot 2025-03-07 113159](https://github.com/user-attachments/assets/0e2677f1-d184-4127-a61e-1d796eacbeb6)
### DATA COLUMNS
![Screenshot 2025-03-07 113401](https://github.com/user-attachments/assets/e39ab7c7-5ce0-4939-a991-34adc9027bb0)
### NULL VALUES
![Screenshot 2025-03-07 113410](https://github.com/user-attachments/assets/0f626212-d9a9-45c4-9b6b-efad825be39f)
### NORMALIZATION
![Screenshot 2025-03-07 114429](https://github.com/user-attachments/assets/ae2ccf47-13b3-4eaa-be73-512474859aeb)
### DATA SPLITING
![Screenshot 2025-03-07 114615](https://github.com/user-attachments/assets/6bb240db-f68c-4415-921b-bd702ca94a28)
### TRAINING & TEST DATA 
![Screenshot 2025-03-07 114728](https://github.com/user-attachments/assets/78bf72ca-1118-4a6e-844c-72694f6405ae)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


