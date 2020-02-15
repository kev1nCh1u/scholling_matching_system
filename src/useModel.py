import pickle
import numpy as np

level_enc = pickle.load(open('model/level_enc_model.sav', 'rb'))
activitve_enc = pickle.load(open('model/activitve_enc_model.sav', 'rb'))
study_enc = pickle.load(open('model/study_enc_model.sav', 'rb'))
biography_enc = pickle.load(open('model/biography_enc_model.sav', 'rb'))
treeModel = pickle.load(open('model/tree_model.sav', 'rb'))

while 1:
    # 使用者輸入資料
    ctData = np.full((1,8), None)
    print()
    print('pleace enter your score...')
    data_column = ['Level of Study', 'GPA', 'TOEIC', 'HSK', 'TOCFL', 'Activitve Certificate', 'Study Plan', 'Autobiography']
    for i in range(len(data_column)):
        ctData[0,i] = input(data_column[i]+ ' : ')
    print(ctData)

    # encoding
    '''
    for i in [0,5,6,7]:
        ctData[:,i] = data_enc[i].transform(ctData[:,i]) # str to int
    '''
    try:
        ctData[:,0] = level_enc.transform(ctData[:,0])
    except:
        print("!!! level not right !!!")
        continue
    try:
        ctData[:,5] = activitve_enc.transform(ctData[:,5])
    except:
        print("!!! activitve not right !!!")
        continue
    try:
        ctData[:,6] = study_enc.transform(ctData[:,6])
    except:
        print("!!! study not right !!!")
        continue
    try:
        ctData[:,7] = biography_enc.transform(ctData[:,7])
    except:
        print("!!! biography not right !!!")
        continue
    #print(ctData)

    # 即時預測
    # 測試資料
    try:
        ctresult = treeModel.predict(ctData)
    except:
        print("!!! error !!!")
        continue
    print('School ranking: Top', ctresult[0])
    print()

