import numpy as np
import matplotlib as plt
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer as SI
from sklearn.linear_model import LinearRegression as LR 
from sklearn.preprocessing import LabelEncoder as LE, OneHotEncoder as OHE

# read
input_data = pd.read_csv('data analytics 154.csv',encoding="utf-8")
data = input_data.iloc[:,[0,6,7,8,9,11,12,13]].values
label = input_data.iloc[:,1].values

# Missing Value nan
mv = SI(missing_values=np.NaN, strategy='median')
mv.fit(data[:,2:5])
data[:,2:5] = mv.transform(data[:,2:5])

print(input_data)
print(pd.DataFrame(data))
print(pd.DataFrame(label))

# str to value
data_enc = LE()
for i in [0,5,6,7]:
    data[:,i] = data_enc.fit_transform(data[:,i])
ohe_enc = OHE(categorical_features=[0])
#data = ohe_enc.fit_transform(data).toarray()
#data = data.astype(int)
print(pd.DataFrame(data))

label_enc = LE()
label = label_enc.fit_transform(label)
print(pd.DataFrame(label))