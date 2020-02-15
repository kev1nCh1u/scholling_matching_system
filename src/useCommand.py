import pickle
import numpy as np
import argparse

def buildArgparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", help="increase Level of Study")
    parser.add_argument("--gpa", help="increase GPA", type=float)
    parser.add_argument("--toeic", help="increase TOEIC", type=int)
    parser.add_argument("--hsk", help="increase HSK", type=int)
    parser.add_argument("--tocfl", help="increase TOCFL", type=int)
    parser.add_argument("--activitve", help="increase Activitve Certificate")
    parser.add_argument("--study", help="increase Study Plan")
    parser.add_argument("--autobiography", help="increase Autobiography")
    return parser
    # python useCommand.py --level master --gpa 7.5 --toeic 570 --hsk 5 --tocfl 3 --activitve no --study b --autobiography b 

def loadModel():
    level_enc = pickle.load(open('model/level_enc_model.sav', 'rb'))
    activitve_enc = pickle.load(open('model/activitve_enc_model.sav', 'rb'))
    study_enc = pickle.load(open('model/study_enc_model.sav', 'rb'))
    biography_enc = pickle.load(open('model/biography_enc_model.sav', 'rb'))
    treeModel = pickle.load(open('model/tree_model.sav', 'rb'))
    return level_enc, activitve_enc, study_enc, biography_enc, treeModel

def main():
    args = buildArgparser().parse_args()
    level_enc, activitve_enc, study_enc, biography_enc, treeModel = loadModel()

    ctData = np.full((1,8), None)
    ctData[0] = [args.level, args.gpa, args.toeic, args.hsk, args.tocfl, args.activitve, args.study, args.autobiography]

    try:
        ctData[:,0] = level_enc.transform(ctData[:,0])
    except:
        print("!!! level not right !!!")
        
    try:
        ctData[:,5] = activitve_enc.transform(ctData[:,5])
    except:
        print("!!! activitve not right !!!")
        
    try:
        ctData[:,6] = study_enc.transform(ctData[:,6])
    except:
        print("!!! study not right !!!")
        
    try:
        ctData[:,7] = biography_enc.transform(ctData[:,7])
    except:
        print("!!! biography not right !!!")
        
    #print(ctData)

    # 即時預測
    # 測試資料
    try:
        ctresult = treeModel.predict(ctData)
        print('School ranking: Top', ctresult[0])
        print()
    except:
        print("!!! error !!!")


main()