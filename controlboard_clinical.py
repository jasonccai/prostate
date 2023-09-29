import nibabel as nb
import os
import shutil
import numpy as np
import numpy.ma as ma
from tqdm import tqdm
import pandas as pd
import copy
import math
from datetime import datetime
from matplotlib import pyplot as plt
from scipy import ndimage
from sklearn import preprocessing
from skimage.measure import regionprops
from skimage.measure import label as sklabel
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.image import random_shear
from tensorflow.keras import callbacks
from tensorflow.data import Dataset
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.metrics import TruePositives, FalsePositives, TrueNegatives, FalseNegatives, Precision, Recall, AUC, Accuracy
import cnn
import cnn_clinical

traincsv = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ztr.csv")
validcsv = os.path.join(os.path.dirname(os.path.realpath(__file__)), "zva.csv")
testscsv = os.path.join(os.path.dirname(os.path.realpath(__file__)), "zte.csv")
    
def preprocess(csv,mode):
    
    def balancecsv(df):
        print("rebalancing dataset...")
        df1 = df[df["Final_Dx"] == 1].reset_index()
        df0 = df[df["Final_Dx"] == 0]
        ind = np.random.choice(len(df1),size=(len(df0)-len(df1)))
        df = pd.concat([df1.iloc[ind],df1,df0],ignore_index=True)
        del df['index']
        df = df.sample(frac=1).reset_index(drop=True)
        return df
    def standardizePSA(df):
        df = np.log10(df)
        mean = df.mean()
        std = df.std()
        df = (df-mean)/std
        return df
    def standardizeage(df):
        mean = df.mean()
        std = df.std()
        df = (df-mean)/std
        return df
    
    df = pd.read_csv(csv).reset_index()
    df = df[df['PSA_nan']!=0]
    if mode == 'tr':
        df = balancecsv(df)

    PSAraw = np.array(standardizePSA(df["PSA"])).astype(np.float32).reshape(-1,1)
    PSATZV = np.array(standardizePSA(df["PSA_TZ"])).astype(np.float32).reshape(-1,1)
    PSAWGV = np.array(standardizePSA(df["PSA_WG"])).astype(np.float32).reshape(-1,1)
    age = np.array(standardizeage(df["age"])).astype(np.float32).reshape(-1,1)
    output = np.array(df["Final_Dx"]).astype(np.int16).reshape(-1,1)

    return df, PSAraw, PSATZV, PSAWGV, age, output

if __name__ == "__main__":
    mode = "p"
    
    if mode == "t":
        _, PSAraw, PSATZV, PSAWGV, age, output = preprocess(traincsv,'tr')
        _, PSArawV, PSATZVV, PSAWGVV, ageV, outputV = preprocess(validcsv,'va')
    
        inputs = np.concatenate((PSAraw, PSATZV, PSAWGV),axis=-1)
        inputsV = np.concatenate((PSArawV, PSATZVV, PSAWGVV),axis=-1)
        
        savedir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results", datetime.now().strftime("%b%d_%Hh%Mm%Ss")+"_CLINICAL_TRAIN")
        checkpointPath = os.path.join(savedir, "Epoch_{epoch:02d}.hdf5")
        bestmodel = callbacks.ModelCheckpoint(checkpointPath, verbose=1, period=1, monitor='val_loss', save_best_only=True, mode='auto', save_weights_only=False)
        model = cnn_clinical.dnn_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=[tf.keras.metrics.BinaryAccuracy()])
        history = model.fit(inputs, output, epochs=110, validation_data=(inputsV, outputV),callbacks=[bestmodel])
        
        result = history.history 
        
        plt.plot(result["loss"],label="loss")
        plt.plot(result["val_loss"])
        plt.show()
        
        plt.plot(result["binary_accuracy"],label="binary_accuracy")
        plt.plot(result["val_binary_accuracy"])
        plt.show()

    if mode == "p":
        from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
        import pickle
        from scipy.interpolate import interp1d

        savedir = "/home/m203898/Videos/clinical_"
        radstpr = 0.909
        weights = "/research/projects/jason/Prostate/1_DenseNet/results/Nov02_18h22m58s_CLINICAL_TRAIN/Epoch_14.hdf5"
        model = load_model(weights)
        dfTE, PSArawT, PSATZVT, PSAWGVT, ageT, outputT = preprocess(testscsv,'te')
        inputsT = np.concatenate((PSArawT, PSATZVT, PSAWGVT),axis=-1)
        
        result = model.predict(inputsT)
        resultc = copy.deepcopy(result)
        ###### 2. Calculate ROC ######
        fpr, tpr, thresholds = roc_curve(outputT, result)
        bootstraplst = []
        for i in range(250):
            indices = np.random.randint(0,len(result),len(result))
            score = roc_auc_score(outputT[indices], result[indices])
            bootstraplst.append(score)
        bootstraplst = sorted(np.array(bootstraplst))
        auc_lower = bootstraplst[int(0.025 * len(bootstraplst))]
        auc_upper = bootstraplst[int(0.975 * len(bootstraplst))]
        resultdct = {}
        resultdct["auc"] = roc_auc_score(outputT, result)
        resultdct["auc_lower_CI"] = auc_lower
        resultdct["auc_upper_CI"] = auc_upper
        ###### 3. Calculate Youden Statistic ######
        radsfpr = interp1d(tpr, fpr)(radstpr)
        # youdenJ = tpr-fpr
        # index = np.argmax(youdenJ)
        # thresholdOpt = round(thresholds[index],ndigits = 4) # 0.3865 path, 0.6 all
        ######
        # gmean = np.sqrt(tpr * (1 - fpr))
        # youdenJOpt = round(gmean[index], ndigits = 4)
        # print('Best Threshold: {} with Youden J statistic: {}'.format(thresholdOpt, youdenJOpt))
        ######
        thresholdOpt = 0.6
        index = (np.abs(thresholds-thresholdOpt)).argmin()
        resultdct["Optimal FPR"] = round(fpr[index], ndigits = 4)
        resultdct["Optimal TPR"] = round(tpr[index], ndigits = 4)
        resultdct["Optimal Threshold"] = thresholdOpt
        resultdct["PIRADS >=3 FPR"] = round(float(radsfpr), ndigits = 4)
        resultdct["PIRADS >=3 TPR"] = round(float(radstpr), ndigits = 4)
        ###### 4. Calculate Metrics ######
        result[result>=thresholdOpt]=1
        result[result<thresholdOpt]=0
        cm = confusion_matrix(result,outputT)
        resultdct["TP"] = cm[1,1]
        resultdct["TN"] = cm[0,0]
        resultdct["FP"] = cm[0,1]
        resultdct["FN"] = cm[1,0]
        resultdct["accuracy"] = accuracy_score(outputT,result)
        resultdct["precision"] = precision_score(outputT,result)
        resultdct["recall"] = recall_score(outputT,result)
        resultdct["f1"] = f1_score(outputT,result)
        # print("image model evaluation...")
        # imagemodel.compile(metrics=temetrics(thresholdOpt))
        # ResultDct = imagemodel.evaluate(datasetTE, batch_size=batch_size, verbose=1, return_dict=True)
        # resultdct = {**ResultDct, **resultdct}
        ###### 5A. Plot CM ######
        fig, ax = plt.subplots(dpi=200)
        ax.matshow(cm, cmap='PuRd')
        ax.set_ylabel('True Label') ; ax.set_xlabel('Predicted Label')
        for (i, j), z in np.ndenumerate(cm):
            ax.text(j, i, z, ha='center', va='center')
        # fig = plt.gcf() ; fig.savefig(savedir+"_CM.png") ; plt.show()
        plt.show()
        plt.clf()
        ###### 5B. Plot ROC ######    
        plt.figure(dpi=200)
        plt.scatter(fpr[index], tpr[index], marker='o', color='red', label='Best', zorder=200)
        plt.scatter(radsfpr, radstpr, marker='o', color='orange', label='Rads',zorder=200)
        plt.plot(fpr, tpr, marker='.') ; plt.plot([0,1],[0,1],linestyle='--', color="orange", zorder=2)
        plt.axis("square")
        plt.xlabel('False Positive Rate') ; plt.ylabel('True Positive Rate')
        string = ""
        for stat,value in resultdct.items():
            string += stat+": "+"%.3f"%value+"\n"
        plt.text(-0.1,-1.2,string)
        fig = plt.gcf() ; fig.savefig(savedir+"_ROC.png",bbox_inches='tight') ; plt.show()
        plt.show()
        plt.clf()
        ###### 5X. Rad Results ######
        radresult = np.array(dfTE["PIRADS"])  # 2,3,4,5
        radresult = np.where(radresult==2,0,radresult) 
        radresult = np.where(radresult==3,1/3,radresult) 
        radresult = np.where(radresult==4,2/3,radresult) 
        radresult = np.where(radresult==5,1,radresult)
        ###### 6. Save ######
        with open(savedir+"_pickle.dat","wb+") as f:
            pickle.dump([outputT,resultc,resultdct,cm,fpr,tpr,index,radresult],f)
        dfTE["final_preds"] = result
        dfTE.to_csv(savedir+".csv")
