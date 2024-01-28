# from matplotlib import pyplot as plt
# for i in range(16):
#     plt.figure()
#     plt.imshow(imgdata[...,i],cmap="gray")
#     plt.show()
#     plt.clf()

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
from skimage.measure import regionprops
from skimage.measure import label as sklabel
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.image import random_shear
from tensorflow.data import Dataset
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.metrics import TruePositives, FalsePositives, TrueNegatives, FalseNegatives, Precision, Recall, AUC, Accuracy
import cnn

mode = "p"        # s(sort), st(sort then train), t(train), p(predict), g(GRAD-CAM)
name = "DCEnew_nodropout_GN"
weightsT = ""
weightsPimage = "/newresearch/research/projects/jason/Prostate/1_DenseNet/results/Oct30_22h22m36s_DCEnew_TRAIN/Epoch_29<-.hdf5"
weightsPclinc = "/newresearch/research/projects/jason/Prostate/1_DenseNet/results/Nov02_18h22m58s_CLINICAL_TRAIN/Epoch_14.hdf5"
valid = True
includeclinical = True
external = False
if external == True:
    includeclinical = False
os.environ['CUDA_VISIBLE_DEVICES']="0"
lr = 1e-4
epochs = 1000
batch_size = 16
classes = 1
shape = (112,96,26)
radstpr = 0.929 if external == True else 0.909
##################
augment = True
lr_flip = True
rotate = math.radians(7) # input degrees. 0 turns off rotation.
shear = 4
deform = 2.5 # 0 turns off deform
jitter = True
gamma = 0 # keep at 0, gamma does not work with images containing negative values
if deform != 0:
    import elasticdeform.tf as etf 
##################

trmetrics=[AUC(name="auc"),'accuracy']
# def temetrics(thresholdOpt):
#     return [TruePositives(name="TP",thresholds=thresholdOpt),FalsePositives(name="FP",thresholds=thresholdOpt),
#             TrueNegatives(name="TN",thresholds=thresholdOpt),FalseNegatives(name="FN",thresholds=thresholdOpt),
#             Precision(name="precision",thresholds=thresholdOpt),Recall(name="recall",thresholds=thresholdOpt)]
if external:
    filepathI = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ImagesX")
    filepathi = os.path.join(os.path.dirname(os.path.realpath(__file__)), "imagesX")
    filepathL = os.path.join(os.path.dirname(os.path.realpath(__file__)), "LabelsX")
    filepathl = os.path.join(os.path.dirname(os.path.realpath(__file__)), "labelsX")
    traincsv = None
    validcsv = None
    testscsv = os.path.join(os.path.dirname(os.path.realpath(__file__)), "zteX.csv")
else:
    filepathI = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Images")
    filepathi = os.path.join(os.path.dirname(os.path.realpath(__file__)), "images")
    filepathL = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Labels")
    filepathl = os.path.join(os.path.dirname(os.path.realpath(__file__)), "labels")
    traincsv = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ztr.csv")
    validcsv = os.path.join(os.path.dirname(os.path.realpath(__file__)), "zva.csv")
    testscsv = os.path.join(os.path.dirname(os.path.realpath(__file__)), "zte.csv")
if mode == "t" or mode == "st":
    savedir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results", datetime.now().strftime("%b%d_%Hh%Mm%Ss")+"_"+name+"_TRAIN")
    os.mkdir(savedir)
if mode == "p":
    savedir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results", datetime.now().strftime("%b%d_%Hh%Mm%Ss")+"_PREDICT")

###################################################################################################

# import matplotlib as mpl ; mpl.rcParams['figure.dpi'] = 300 ; plt.figure() ; plt.hist(image.flatten(),bins=100) ; plt.show() ; plt.clf()
def normalize(image,mask,each):
    if not each.startswith("ADC"):
        # 1. standardize
        mean = image[mask==1].mean()
        std = image[mask==1].std()
        image = (image-mean)/std
        # 2. clip
        lower, upper = np.percentile(image,(0.3,99.7))
        image = np.clip(image,lower,upper)
        # 3. normalize
        # maxx, minn = image.max(), image.min()
        # image = (image-minn)/(maxx-minn)
    else:
        image = np.clip(image,0,2000)
        image = image/2000  
    return image

def process_scan(each,filepathI,filepathL,filepathi,filepathl):
    image = nb.load(os.path.join(filepathI,each)).get_fdata().astype("float32")
    mask = nb.load(os.path.join(filepathL,each)).get_fdata().astype("float32")
    image = normalize(image,mask,each)
    np.save(os.path.join(filepathi,each[:-7]+".npy"),image)
    np.save(os.path.join(filepathl,each[:-7]+".npy"),mask)

###################################################################################################

if mode == "s" or mode == "st":
    selection = input("sort and redo?")
    if selection == "y":
        print("Deleting...")
        shutil.rmtree(filepathi)
        os.mkdir(filepathi)
        print("Deleted...")
    assert sorted(os.listdir(filepathI)) == sorted(os.listdir(filepathL)), "Image and Label folders are not the same"

    for each in tqdm(os.listdir(filepathI)):
        process_scan(each,filepathI,filepathL,filepathi,filepathl)
           
###################################################################################################

def generator(df,filepathi,mode,augment,shear,jitter):

    def balancecsv(df):
        print("rebalancing dataset...")
        
        ### undersample the overrepresented class ###
        df1 = df[df["Final_Dx"] == 1]
        df0 = df[df["Final_Dx"] == 0].reset_index()
        ind = np.random.choice(len(df0),size=len(df1),replace=False)
        df0 = df0[df0.index.isin(ind)]
        df = pd.concat([df0,df1],ignore_index=True)
        del df['index']
        df = df.sample(frac=1).reset_index(drop=True)
        return df

        ### oversample the underrepresented class ###
        # df1 = df[df["Final_Dx"] == 1].reset_index()
        # df0 = df[df["Final_Dx"] == 0]
        # ind = np.random.choice(len(df1),size=(len(df0)-len(df1)))
        # df = pd.concat([df1.iloc[ind],df1,df0],ignore_index=True)
        # del df['index']
        # df = df.sample(frac=1).reset_index(drop=True)
        # return df
    if mode == "train" or mode == "debug":
        df = balancecsv(df)
        # pass
        
    def jitterNreshape(img,lab,upperPx,lowerPx,upperPy,lowerPy,upperPz):
        connectivity = sklabel(lab).astype("uint16")
        properties = regionprops(connectivity)
        bbox = properties[0].bbox # right, posterior, caudal, left, superior, cranial
        
        def getbounds(dimlower,dimupper,upperP,lowerP):
            topb = img.shape[dimlower]-bbox[dimupper]
            lowb = bbox[dimlower]
            topb = int(topb*upperP)
            lowb = int(lowb*lowerP)
            topb = bbox[dimupper]+topb
            lowb = bbox[dimlower]-lowb    
            return topb,lowb
        
        xtopb,xlowb = getbounds(0,3,upperPx,lowerPx)
        ytopb,ylowb = getbounds(1,4,upperPy,lowerPy)
        ztopb = bbox[5]+upperPz ; zlowb = bbox[2]-1
        if ztopb>img.shape[2]:
            ztopb = bbox[5]
        if zlowb<0:
            zlowb = bbox[2]
    
        img = img[xlowb:xtopb,ylowb:ytopb,zlowb:ztopb]
        x = shape[0] / img.shape[0]
        y = shape[1] / img.shape[1]
        z = shape[2] / img.shape[2]
        img = ndimage.zoom(img,(x,y,z),order=1,mode='reflect')
        return img

    for i,row in df.iterrows():
        T2W = np.load(os.path.join(filepathi,row["T2W_filename"]))
        ADC = np.load(os.path.join(filepathi,row["ADC_filename"]))
        DWI = np.load(os.path.join(filepathi,row["DWI_filename"]))
        DCE000 = np.load(os.path.join(filepathi,row["DCE000_filename"]))
        DCE010 = np.load(os.path.join(filepathi,row["DCE010_filename"]))
        DCE100 = np.load(os.path.join(filepathi,row["DCE100_filename"]))
        T2WL = np.load(os.path.join(filepathl,row["T2W_filename"]))
        ADCL = np.load(os.path.join(filepathl,row["ADC_filename"]))
        DWIL = np.load(os.path.join(filepathl,row["DWI_filename"]))
        DCEL = np.load(os.path.join(filepathl,row["DCE000_filename"]))
        
        if augment and jitter and (mode=="train" or mode=="debug"):
            upperPx,lowerPx,upperPy,lowerPy,upperPz = np.random.uniform(),np.random.uniform(),np.random.uniform(),np.random.uniform(),np.random.randint(1,2)
            T2W,ADC,DWI = jitterNreshape(T2W,T2WL,upperPx,lowerPx,upperPy,lowerPy,upperPz),jitterNreshape(ADC,ADCL,upperPx,lowerPx,upperPy,lowerPy,upperPz),jitterNreshape(DWI,DWIL,upperPx,lowerPx,upperPy,lowerPy,upperPz)
            DCE000,DCE010,DCE100 = jitterNreshape(DCE000,DCEL,upperPx,lowerPx,upperPy,lowerPy,upperPz),jitterNreshape(DCE010,DCEL,upperPx,lowerPx,upperPy,lowerPy,upperPz),jitterNreshape(DCE100,DCEL,upperPx,lowerPx,upperPy,lowerPy,upperPz)
        else:
            T2W,ADC,DWI = jitterNreshape(T2W,T2WL,0.3,0.3,0.3,0.3,1),jitterNreshape(ADC,ADCL,0.3,0.3,0.3,0.3,1),jitterNreshape(DWI,DWIL,0.3,0.3,0.3,0.3,1)
            DCE000,DCE010,DCE100 = jitterNreshape(DCE000,DCEL,0.3,0.3,0.3,0.3,1),jitterNreshape(DCE010,DCEL,0.3,0.3,0.3,0.3,1),jitterNreshape(DCE100,DCEL,0.3,0.3,0.3,0.3,1)
        if augment and shear!=0 and (mode=="train" or mode=="debug"):
            shearfactorx = np.random.uniform(-shear,shear); shearfactory = shearfactorx*np.random.uniform(0.75,1.25)*-1
            T2W,ADC,DWI = random_shear(T2W,shearfactorx,1,0,2,'reflect',interpolation_order=1),random_shear(ADC,shearfactorx,1,0,2,'reflect',interpolation_order=1),random_shear(DWI,shearfactorx,1,0,2,'reflect',interpolation_order=1)
            DCE000,DCE010,DCE100 = random_shear(DCE000,shearfactorx,1,0,2,'reflect',interpolation_order=1),random_shear(DCE010,shearfactorx,1,0,2,'reflect',interpolation_order=1),random_shear(DCE100,shearfactorx,1,0,2,'reflect',interpolation_order=1)
            T2W,ADC,DWI,DCE000,DCE010,DCE100 = np.rot90(T2W),np.rot90(ADC),np.rot90(DWI),np.rot90(DCE000),np.rot90(DCE010),np.rot90(DCE100)
            T2W,ADC,DWI = random_shear(T2W,shearfactory,1,0,2,'reflect',interpolation_order=1),random_shear(ADC,shearfactory,1,0,2,'reflect',interpolation_order=1),random_shear(DWI,shearfactory,1,0,2,'reflect',interpolation_order=1)
            DCE000,DCE010,DCE100 = random_shear(DCE000,shearfactory,1,0,2,'reflect',interpolation_order=1),random_shear(DCE010,shearfactory,1,0,2,'reflect',interpolation_order=1),random_shear(DCE100,shearfactory,1,0,2,'reflect',interpolation_order=1)
            T2W,ADC,DWI,DCE000,DCE010,DCE100 = np.rot90(T2W,3),np.rot90(ADC,3),np.rot90(DWI,3),np.rot90(DCE000,3),np.rot90(DCE010,3),np.rot90(DCE100,3)
            
        T2W,ADC,DWI,DCE000,DCE010,DCE100 = tf.cast(T2W,tf.float32),tf.cast(ADC,tf.float32),tf.cast(DWI,tf.float32),tf.cast(DCE000,tf.float32),tf.cast(DCE010,tf.float32),tf.cast(DCE100,tf.float32)
        DCE000,DCE010,DCE100 = tf.expand_dims(DCE000,axis=-1),tf.expand_dims(DCE010,axis=-1),tf.expand_dims(DCE100,axis=-1)
        DCE = tf.concat([DCE000,DCE010,DCE100],axis=-1)
 
        label = np.array(row["Final_Dx"]).reshape(-1)
        label = tf.cast(label, tf.int16)
        
        if mode == "debug" or mode == "grad":
            yield {"T2W":T2W,"ADC":ADC,"DWI":DWI,"DCE":DCE}, label, row["T2W_filename"]
        else:
            yield {"T2W":T2W,"ADC":ADC,"DWI":DWI,"DCE":DCE}, label

def plot(dct,label,filename=None,ind=0,probability=None,heatmap=None,threshold=0.1):
    if ind != None:
        T2W = dct["T2W"][ind]; ADC = dct["ADC"][ind]; DWI = dct["DWI"][ind]; DCE = dct["DCE"][ind]
    else:
        T2W = dct["T2W"]; ADC = dct["ADC"]; DWI = dct["DWI"]; DCE = dct["DCE"]
    if isinstance(heatmap,np.ndarray):
        heatmap[heatmap<threshold]=0
        heatmap = np.rot90(heatmap)
    else:
        heatmap = np.zeros((shape[1],shape[0],shape[2]))
    for i in np.linspace(0,shape[2]-1,shape[2]-1):
        i = int(i)
        plt.figure(dpi=300)
        ax = plt.subplot(231)
        plt.imshow(np.rot90(T2W[...,i].numpy()),cmap="gray");plt.axis('off');ax.set_title("T2")
        heatmap = ma.masked_where(heatmap == 0, heatmap)
        # plt.imshow(heatmap[...,i],"cool",alpha = 0.35)
        ax = plt.subplot(232)
        plt.imshow(np.rot90(ADC[...,i].numpy()),cmap="gray");plt.axis('off');ax.set_title("ADC")
        heatmap = ma.masked_where(heatmap == 0, heatmap)
        # plt.imshow(heatmap[...,i],"cool",alpha = 0.35)
        ax = plt.subplot(233)
        plt.imshow(np.rot90(DWI[...,i].numpy()),cmap="gray");plt.axis('off');ax.set_title("DWI")
        heatmap = ma.masked_where(heatmap == 0, heatmap)
        # plt.imshow(heatmap[...,i],"cool",alpha = 0.35)
        ax = plt.subplot(234)
        plt.imshow(np.rot90(DCE[...,i,0].numpy()),cmap="gray");plt.axis('off');ax.set_title("T1 DCE 0s")
        heatmap = ma.masked_where(heatmap == 0, heatmap)
        # plt.imshow(heatmap[...,i],"cool",alpha = 0.35)
        ax = plt.subplot(235)
        plt.imshow(np.rot90(DCE[...,i,1].numpy()),cmap="gray");plt.axis('off');ax.set_title("T1 DCE 10s")
        heatmap = ma.masked_where(heatmap == 0, heatmap)
        # plt.imshow(heatmap[...,i],"cool",alpha = 0.35)
        ax = plt.subplot(236)
        plt.imshow(np.rot90(DCE[...,i,2].numpy()),cmap="gray");plt.axis('off');ax.set_title("T1 DCE 100s")
        heatmap = ma.masked_where(heatmap == 0, heatmap)
        # plt.imshow(heatmap[...,i],"cool",alpha = 0.35)
        # plt.colorbar()
        if isinstance(filename,tf.Tensor) and isinstance(probability,tf.Tensor):
            plt.suptitle(str(filename[0].numpy())[2:-1]+" | "+str(tf.squeeze(label[ind]).numpy())+" | "+str(tf.squeeze(probability).numpy()))
        elif isinstance(filename,str):
            plt.suptitle(str(filename)+" | "+str(tf.squeeze(label[ind]).numpy()))
        elif isinstance(filename,tf.Tensor):
            plt.suptitle(str(filename[0].numpy())[2:-1]+" | "+str(tf.squeeze(label[ind]).numpy()))
        else:
            plt.suptitle(str(tf.squeeze(label[ind]).numpy()))
        plt.tight_layout(pad=1.7)
        plt.show()
        plt.clf()
    input("Continue?")
        
# dfTR = pd.read_csv(traincsv,index_col=0)
# while True:
#     for dct,label,filename in generator(dfTR,filepathi,"debug",augment,shear,jitter):
#         plot(dct,label,filename,ind=None)

##################################################################################################        

if mode == "t" or mode == "st":
    autotune = tf.data.AUTOTUNE
    dfTR = pd.read_csv(traincsv,index_col=0)
    generatorTR = lambda: generator(dfTR,filepathi,"train",augment,shear,jitter)
    datasetTR = Dataset.from_generator(generatorTR,output_types=(({"T2W":tf.float32,"ADC":tf.float32,"DWI":tf.float32,"DCE":tf.float32},tf.int16)),
                                       output_shapes=(({"T2W":shape,"ADC":shape,"DWI":shape,"DCE":(shape[0],shape[1],shape[2],3)},(1,))))
    
    def augmentor(imagedct,lr_flip,rotate,deform,gamma,shear):
        T2W = imagedct["T2W"]; ADC = imagedct["ADC"]; DWI = imagedct["DWI"]; DCE = imagedct["DCE"]
        DCE000,DCE010,DCE100 = DCE[...,0],DCE[...,1],DCE[...,2]
        if lr_flip and tf.cast(tf.random.uniform((),0,2,tf.int32),tf.bool): # WARNING: TF FLIP UP DOWN FLIPS LEFT RIGHT because the image is loaded 90 degrees rotated clockwise.
            T2W,ADC,DWI = tf.image.flip_up_down(T2W),tf.image.flip_up_down(ADC),tf.image.flip_up_down(DWI)
            DCE000,DCE010,DCE100 = tf.image.flip_up_down(DCE000),tf.image.flip_up_down(DCE010),tf.image.flip_up_down(DCE100)
        if rotate!=0:
            rot = tf.random.uniform((),-rotate,rotate)
            T2W,ADC,DWI = tfa.image.rotate(T2W,rot,fill_mode="reflect"),tfa.image.rotate(ADC,rot,fill_mode="reflect"),tfa.image.rotate(DWI,rot,fill_mode="reflect")
            DCE000,DCE010,DCE100 = tfa.image.rotate(DCE000,rot,fill_mode="reflect"),tfa.image.rotate(DCE010,rot,fill_mode="reflect"),tfa.image.rotate(DCE100,rot,fill_mode="reflect")
        if deform!=0:
            deform = tf.random.uniform((),1,deform)
            displacement = tf.math.multiply(tf.random.normal((3,2,2,2)),deform)
            T2W,ADC,DWI = etf.deform_grid(T2W,displacement=displacement,order=1,mode='nearest'),etf.deform_grid(ADC,displacement=displacement,order=1,mode='nearest'),etf.deform_grid(DWI,displacement=displacement,order=1,mode='nearest')
            DCE000,DCE010,DCE100 = etf.deform_grid(DCE000,displacement=displacement,order=1,mode='nearest'),etf.deform_grid(DCE010,displacement=displacement,order=1,mode='nearest'),etf.deform_grid(DCE100,displacement=displacement,order=1,mode='nearest')
        if gamma!=0:
            gam = tf.random.uniform((),1-gamma,1+gamma)
            T2W,DWI = tf.image.adjust_gamma(T2W,gam),tf.image.adjust_gamma(DWI,gam)
            DCE000,DCE010,DCE100 = tf.image.adjust_gamma(DCE000,gam),tf.image.adjust_gamma(DCE010,gam),tf.image.adjust_gamma(DCE100,gam)
        DCE000,DCE010,DCE100 = tf.expand_dims(DCE000,axis=-1),tf.expand_dims(DCE010,axis=-1),tf.expand_dims(DCE100,axis=-1)
        DCE = tf.concat([DCE000,DCE010,DCE100],axis=-1)
        imagedct["T2W"] = T2W ; imagedct["ADC"] = ADC ; imagedct["DWI"] = DWI ; imagedct["DCE"] = DCE
        return imagedct

    if augment:
        datasetTR = datasetTR.map(lambda x,y: (augmentor(x,lr_flip,rotate,deform,gamma,shear),y), num_parallel_calls=autotune)
    datasetTR = datasetTR.batch(batch_size).prefetch(buffer_size=autotune)

    dfVA = pd.read_csv(validcsv,index_col=0)
    generatorVA = lambda: generator(dfVA,filepathi,"valid",augment,shear,jitter)
    datasetVA = Dataset.from_generator(generatorVA,output_types=(({"T2W":tf.float32,"ADC":tf.float32,"DWI":tf.float32,"DCE":tf.float32},tf.int16)),
                                       output_shapes=(({"T2W":shape,"ADC":shape,"DWI":shape,"DCE":(shape[0],shape[1],shape[2],3)},(1,))))
    datasetVA = datasetVA.batch(batch_size).prefetch(buffer_size=autotune)
    
    # while True:
    #     for dct, label in datasetTR.take(200):
    #         plot(dct,label,None,ind=0,probability=None,heatmap=None,threshold=0.1)

    if weightsT:
        print("loading weights...")
        model = load_model(weightsT)
    else:
        model = cnn.cnn_model(shape=shape,classes=classes)
        model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=trmetrics)
    model.summary(line_length=150)
    
    from tensorflow.keras import callbacks
    csv_logger = callbacks.CSVLogger(os.path.join(savedir, "Results.csv"))
    checkpointPath = os.path.join(savedir, "Epoch_{epoch:02d}.hdf5")
    best_model = callbacks.ModelCheckpoint(checkpointPath, verbose=1, period=1, monitor='loss', save_best_only=True, mode='auto', save_weights_only=False)
    lr_reducer = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.7, patience=30, verbose=1, mode='auto', min_delta=0.003, cooldown=10, min_lr=1e-7)
    # tensorboard = callbacks.TensorBoard(log_dir=savedir, histogram_freq=0)
            
    if valid == True:
        model.fit(datasetTR,validation_data=datasetVA,epochs=epochs,shuffle=False,verbose=1,callbacks=[best_model,lr_reducer,csv_logger])
    else:
        model.fit(datasetTR,epochs=epochs,shuffle=False,verbose=1,callbacks=[best_model,lr_reducer,csv_logger])
        
###################################################################################################

if mode == "p":
    from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
    import pickle
    import controlboard_clinical
    from scipy.interpolate import interp1d

    if includeclinical:
        _, PSArawT, PSATZVT, PSAWGVT, age, outputT = controlboard_clinical.preprocess(testscsv,traincsv,'te')
        inputsT = np.concatenate((PSArawT, PSATZVT, PSAWGVT),axis=-1)
    dfTE = pd.read_csv(testscsv,index_col=0).reset_index().drop('index', 1)
    GT = np.array(dfTE["Final_Dx"])
    generatorTE = lambda: generator(dfTE,filepathi,"test",augment,shear,jitter)
    datasetTE = Dataset.from_generator(generatorTE,output_types=(({"T2W":tf.float32,"ADC":tf.float32,"DWI":tf.float32,"DCE":tf.float32},tf.int16)),
                                       output_shapes=(({"T2W":shape,"ADC":shape,"DWI":shape,"DCE":(shape[0],shape[1],shape[2],3)},(1,))))
    datasetTE = datasetTE.batch(1)
    imagemodel = load_model(weightsPimage)
    clincmodel = load_model(weightsPclinc)

    ###### 1. Predict ######
    print("image model prediction...")
    # imageresult = imagemodel.predict(datasetTE, batch_size=batch_size, verbose=1)
    # imageresult = imageresult[:,0]
    # imageresult = np.random.random(len(dfTE))
    with open("/newresearch/research/projects/jason/Prostate/1_DenseNetOld/results/Jan23_13h20m34s_PREDICT_pickle_COMBINED400new.dat", "rb") as f:
        GT,imageresult,resultdct,cm,fpr,tpr,index = pickle.load(f)
    dfTE["image_preds"] = imageresult
    result = imageresult
    
    if includeclinical:
        print("clinical model prediction...")
        clincresult = clincmodel.predict(inputsT, verbose=1)
        clincresult = clincresult[:,0]
        dfTE.loc[dfTE['PSA_nan']!=0,"clinical_preds"] = clincresult
        ##### 1X. Merge #####
        dfTE['composite_preds'] = 0.2*dfTE["clinical_preds"]+0.8*dfTE["image_preds"]
        # x = dfTE[(dfTE.image_preds < 0.65) & (dfTE.image_preds > 0.35)]
        # x = x[(x.clinical_preds > 0.65) | (x.clinical_preds < 0.35)]
        # x.composite_preds = x.clinical_preds
        # dfTE.loc[x.index, :] = x[:]
        ##################  
        dfTE.composite_preds.fillna(dfTE.image_preds, inplace=True)
        result = copy.deepcopy(dfTE['composite_preds'])
        resultc = copy.deepcopy(result)
    else:
        resultc = copy.deepcopy(imageresult)
    ###### 2. Calculate ROC ######
    fpr, tpr, thresholds = roc_curve(GT, result)
    bootstraplst = []
    for i in range(250):
        indices = np.random.randint(0,len(result),len(result))
        score = roc_auc_score(GT[indices], result[indices])
        bootstraplst.append(score)
    bootstraplst = sorted(np.array(bootstraplst))
    auc_lower = bootstraplst[int(0.025 * len(bootstraplst))]
    auc_upper = bootstraplst[int(0.975 * len(bootstraplst))]
    resultdct = {}
    resultdct["auc"] = roc_auc_score(GT, result)
    resultdct["auc_lower_CI"] = auc_lower
    resultdct["auc_upper_CI"] = auc_upper
    ###### 2X. Calculate ROC for clinical model ######
    if includeclinical:
        clincauc = roc_auc_score(outputT, clincresult)
        clincfpr, clinctpr, clincthresholds = roc_curve(outputT, clincresult)
    ###### 3. Calculate Youden Statistic ######
    radsfpr = interp1d(tpr, fpr)(radstpr)
    # youdenJ = tpr-fpr
    # index = np.argmax(youdenJ)
    # thresholdOpt = round(thresholds[index],ndigits = 4)
    # 5/0
    # validation datset: 0.3265 (image model path), 0.3038 (image model all), 0.3859 (combined model path), 0.3764 (combined model all)
    #####
    # gmean = np.sqrt(tpr * (1 - fpr))
    # youdenJOpt = round(gmean[index], ndigits = 4)
    # print('Best Threshold: {} with Youden J statistic: {}'.format(thresholdOpt, youdenJOpt))
    ######
    thresholdOpt = 0.3764
    index = (np.abs(thresholds-thresholdOpt)).argmin()
    resultdct["Optimal FPR"] = round(fpr[index], ndigits = 4)
    resultdct["Optimal TPR"] = round(tpr[index], ndigits = 4)
    resultdct["Optimal Threshold"] = thresholdOpt
    resultdct["PIRADS >=3 FPR"] = round(float(radsfpr), ndigits = 4)
    resultdct["PIRADS >=3 TPR"] = round(float(radstpr), ndigits = 4)
    ###### 4. Calculate Metrics ######
    result[result>=thresholdOpt]=1
    result[result<thresholdOpt]=0
    cm = confusion_matrix(result,GT)
    resultdct["TP"] = cm[1,1]
    resultdct["TN"] = cm[0,0]
    resultdct["FP"] = cm[0,1]
    resultdct["FN"] = cm[1,0]
    resultdct["accuracy"] = accuracy_score(GT,result)
    resultdct["precision"] = precision_score(GT,result)
    resultdct["recall"] = recall_score(GT,result)
    resultdct["f1"] = f1_score(GT,result)
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
    fig = plt.gcf() ; fig.savefig(savedir+"_CM.png") ; plt.show()
    plt.clf()
    ###### 5B. Plot ROC ######    
    plt.figure(dpi=200)
    plt.scatter(fpr[index], tpr[index], marker='o', s = 10, color='red', label='Best',zorder=100)
    plt.scatter(radsfpr, radstpr, marker='o', s = 10, color='orange', label='Rads',zorder=200)
    plt.plot(fpr, tpr, marker=',', linewidth=1) ; plt.plot([0,1],[0,1],linestyle='--', linewidth=1, color="orange")
    plt.axis("square")
    plt.xlabel('False Positive Rate') ; plt.ylabel('True Positive Rate')
    string = ""
    for stat,value in resultdct.items():
        string += stat+": "+"%.3f"%value+"\n"
    plt.text(-0.1,-1.2,string)
    fig = plt.gcf()
    fig.savefig(savedir+"_ROC.png",bbox_inches='tight')
    plt.show()
    plt.clf()
    ###### 6. Save ######
    with open(savedir+"_pickle.dat","wb+") as f:
        pickle.dump([GT,resultc,resultdct,cm,fpr,tpr,index],f)
    dfTE["final_preds"] = result
    dfTE.to_csv(savedir+".csv")

###################################################################################################    

if mode == "g":
    dfGR = pd.read_csv(testscsv,index_col=0).reset_index()
    GT = np.array(dfGR["Final_Dx"])
    generatorGR = lambda: generator(dfGR,filepathi,"grad",augment,shear,jitter)
    datasetGR = Dataset.from_generator(generatorGR,output_types=(({"T2W":tf.float32,"ADC":tf.float32,"DWI":tf.float32,"DCE":tf.float32},tf.int16,tf.string)))
    datasetGR = datasetGR.batch(1)
    model = load_model(weightsPimage)
    
    def make_gradcam_heatmap(inputlst,model,ind):
        grad_model = Model([model.inputs],[model.get_layer("gradcam_layer").output, model.output])
        with tf.GradientTape() as tape:
            gradcam_layer,preds = grad_model(inputs=inputlst)
            probability = preds[ind]
        grads = tape.gradient(probability,gradcam_layer)[ind,...]
        pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
        last_conv_layer_output = gradcam_layer[ind]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy(),probability

    for dct,label,filename in datasetGR.take(100):
        if int(label.numpy()) == 1:
            ind=0 # which element of the batch to plot?
            inputlst = [dct["T2W"],dct["ADC"],dct["DWI"],dct["DCE"]]
            heatmap,probability = make_gradcam_heatmap(inputlst,model,ind)
            Idim0,Idim1,Idim2 = heatmap.shape[0],heatmap.shape[1],heatmap.shape[2]
            Odim0,Odim1,Odim2 = shape
            fact0,fact1,fact2 = Odim1/Idim1,Odim0/Idim0,Odim2/Idim2 # because the image is loaded 90 deg rotated clockwise
            heatmap = ndimage.zoom(heatmap,(fact0,fact1,fact2),order=1)
            if probability.numpy() < 0.1:
            # if True:
                plot(dct,label,filename,ind,probability,heatmap=heatmap,threshold=0.1)

# plt.figure(dpi=200)
# plt.plot(clincfpr, clinctpr, marker='.') ; plt.plot([0,1],[0,1],linestyle='--', color="orange")
# plt.xlabel('False Positive Rate') ; plt.ylabel('True Positive Rate')
# plt.axis("square")
# plt.show()
# plt.clf()