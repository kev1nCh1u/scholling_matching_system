import numpy as np
import matplotlib as plt
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer as SI
from sklearn.linear_model import LinearRegression as LR 
from sklearn.preprocessing import LabelEncoder as LE, OneHotEncoder as OHE
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.metrics.classification import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_boston
from sklearn import preprocessing
from subroutine.CMatrix import cm_plot     #副程式
from subroutine.ROC_AUC import acu_curve   #副程式


# read
input_data = pd.read_csv('data\\data analytics 154.csv',encoding="utf-8")
#print(input_data)

#提取CSV檔 標籤
input_data_column = list(input_data.columns.values)
#print(input_data_column)
data_column = []
data_column.append(input_data_column[0])
data_column.extend(input_data_column[6:10])
data_column.extend(input_data_column[11:14])
print(data_column)

#提取需要的資料
data = input_data.iloc[:,[0,6,7,8,9,11,12,13]].values
label = input_data.iloc[:,1].values
print(pd.DataFrame(data))
print(pd.DataFrame(label))

# Missing Value nan
mv = SI(missing_values=np.NaN, strategy='median')
mv.fit(data[:,2:5])
data[:,2:5] = mv.transform(data[:,2:5])
#print(pd.DataFrame(data))
#print(pd.DataFrame(label))

# to lowercase 
for i in range(len(data[:,0])):
    data[i,0] = data[i,0].lower()
#print(data[:,0])

# str to value
#data_enc = LE()
data_enc = np.full(8, None)
for i in [0,5,6,7]:
    data_enc[i] = LE()
    data[:,i] = data_enc[i].fit_transform(data[:,i])
ohe_enc = OHE(categorical_features=[0])
#data = ohe_enc.fit_transform(data).toarray()
#data = data.astype(int)

# 正規化
'''
for i in range(8):
    data[:,i] = preprocessing.scale(data[:,i])
'''
#print(data ,pd.DataFrame(data))

# str get int
for i in range(len(label)):
    if(label[i].find('Top ') != -1):
        label[i] = label[i].replace('Top ', '')
    elif(label[i].find('outside top ') != -1):
        label[i] = label[i].replace('outside top ', '50')
    label[i] = int(label[i])

# str to value
'''
label_enc = LE()
label = label_enc.fit_transform(label)
print(pd.DataFrame(label))
'''
#print('label', label)

# KNN
'''
x = data   #提取權重較重的資料
y = label
x = preprocessing.scale(x)
#print(pd.DataFrame(x))


y_median = np.median(pd.DataFrame(set(y)))      #計算y資料的中位數
for i in range(len(y)):                         #以中位數把y資料切成0,1
    if y[i] > y_median:
        y[i] = 1
    else:
        y[i] = 0
#print(x)        
#print(y)

knn = KNeighborsClassifier(n_neighbors=4)      #用KNN分類法,尋找4個鄰居
cv = ShuffleSplit(n_splits=10, test_size=0.4)   # K為10，將資料分成train,test
scores_clf_svc_cv = cross_val_score(knn, x, y, cv=cv, scoring='accuracy')
print(scores_clf_svc_cv)                        #列印10次的值
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_clf_svc_cv.mean(), scores_clf_svc_cv.std()))  #計算10次的平均模型準確度
print('----------------------------------')

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4)    #將資料分成train,test
knn = KNeighborsClassifier(n_neighbors=4)      #用KNN分類法,尋找4個鄰居
knn.fit(X_train, y_train)                       #資料訓練
y_predict = knn.predict(X_test)                 #訓練出來的資料
y_pred_proba = knn.predict_proba(X_test)[:,1]   #訓練出來的資料 

print("accuracy score:", accuracy_score(y_test, y_predict))     #計算模型準確度
#cm_plot(y_test, y_predict)      #計算混淆矩陣
print('----------------------------------')

acu_curve(y_test, y_pred_proba) #KNN計算真正率 & 假正率
print('----------------------------------')

svm = svm.SVC(kernel='linear', probability=True)
y_score = svm.fit(X_train, y_train).decision_function(X_test)
acu_curve(y_test, y_score)      #SVM計算真正率 & 假正率
print('----------------------------------')
'''
'''
#尋找鄰居最好的K值
#k_range = range(1,50)
#k_scores = []
#for k_number in k_range:
#    knn = KNeighborsClassifier(n_neighbors=k_number)
#    scores = cross_val_score(knn, x, y, cv=10, scoring='accuracy')
#    k_scores.append(scores.mean())
#plt.plot(k_range,k_scores)
#plt.xlabel('Value of K for KNN')
#plt.ylabel('Cross-Validated Accuracy')
#plt.show()
'''

# 訓練並擬合後測試
regs = LR()
regs.fit(data, label)

# 測試資料
result = regs.predict(data)

# 驗證100比
#y_predict = knn.predict(data)
#print(pd.DataFrame(label_enc.inverse_transform(result)))
#print(pd.DataFrame(label_enc.inverse_transform(label)))
#print(pd.DataFrame(result))
#print(pd.DataFrame(label))
labelSqu = label.reshape(-1, 1)
resultSqu = result.reshape(-1, 1)

for i in range(len(label)):
    print(label[i], result[i])
print("score:", r2_score(label, result.round()))    #計算模型準確度

while 1:
    # 使用者輸入資料
    ctData = np.full((1,8), None)
    print()
    print('pleace enter your score...')
    for i in range(len(data_column)):
        ctData[0,i] = input(data_column[i]+ ' : ')
    #print(ctData)
    for i in [0,5,6,7]:
        ctData[:,i] = data_enc[i].transform(ctData[:,i]) # str to int
    #print(ctData)


    # 即時預測
    # 測試資料
    ctresult = regs.predict(ctData)
    print('School ranking: Top', ctresult[0].round())
    print()

