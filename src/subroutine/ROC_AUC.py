import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
def acu_curve(original_label, predict_label):
    fpr,tpr,_ = roc_curve(original_label, predict_label)    #計算真正率 & 假正率
    roc_auc = auc(fpr,tpr)  #計算AUC值 
    print("AUC:", roc_auc)
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc) #真正率為縱座標,假正率為橫坐標
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()