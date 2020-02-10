import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def cm_plot(original_label, predict_label, pic=None):
    cm = confusion_matrix(original_label, predict_label)    #預測資料的混淆矩陣
    print(cm)       #列印混淆矩陣值
    print(classification_report(original_label, predict_label))     #計算Precision / Recall / F-measure

    plt.matshow(cm, cmap=plt.cm.Blues)     #使用藍色cm.Blues畫出混淆矩陣
    plt.colorbar()                  #顏色標籤
    for x in range(len(cm)):        #打印數值到plt上
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
    plt.xlabel('Predicted label')   # X軸標籤
    plt.ylabel('True label')        # Y軸標籤
    plt.title('confusion matrix') 
    plt.show()                      #顯示圖片