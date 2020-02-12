import pickle
import numpy as np

level_enc = pickle.load(open('model/level_enc_model.sav', 'rb'))
activitve_enc = pickle.load(open('model/activitve_enc_model.sav', 'rb'))
study_enc = pickle.load(open('model/study_enc_model.sav', 'rb'))
biography_enc = pickle.load(open('model/biography_enc_model.sav', 'rb'))
treeModel = pickle.load(open('model/tree_model.sav', 'rb'))

while 0:
    # 使用者輸入資料
    ctData = np.full((1,8), None)
    print()
    print('pleace enter your score...')
    data_column = ['Level of Study', 'GPA', 'TOEIC', 'HSK', 'TOCFL', 'Activitve Certificate', 'Study Plan', 'Autobiography']
    for i in range(len(data_column)):
        ctData[0,i] = input(data_column[i]+ ' : ')
    #print(ctData)

    # encoding
    '''
    for i in [0,5,6,7]:
        ctData[:,i] = data_enc[i].transform(ctData[:,i]) # str to int
    '''
    ctData[:,0] = level_enc.transform(ctData[:,0])
    ctData[:,5] = activitve_enc.transform(ctData[:,5])
    ctData[:,6] = study_enc.transform(ctData[:,6])
    ctData[:,7] = biography_enc.transform(ctData[:,7])
    #print(ctData)


    # 即時預測
    # 測試資料
    ctresult = treeModel.predict(ctData)
    print('School ranking: Top', ctresult[0])
    print()

