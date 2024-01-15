"""
    Fault Diagnosis of Separator Systems Done Using DNN Model
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,BatchNormalization
from tensorflow.keras import models
from tensorflow.keras.activations import relu,softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal,Zeros
from sklearn.metrics import accuracy_score,confusion_matrix

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

"""
    Building DNN Model
"""
class MLP():
    def __init__(self,filePath,savePicturePath):
        self.filePath = filePath # Read File Path
        self.dataSet = self.ReadData() # Getting the Training Dataset
        self.classificationNumber = 14 # Number of Classifications
        self.savePicturePath = savePicturePath # Heat Map Save Path
        self.index = self.GetCorrelationCoefficient() # Get the Index of the Correlation Coefficient
        self.newFeature = self.dataSet[self.index] # Search for Relevance Features
        self.target = self.dataSet["Label"] # Get Tagged Values
        self.data_standard = self.Standard()
        self.x_train, self.x_test, self.y_train, self.y_test = self.SplitTrainAndTest()
        self.unit1 = 16
        self.unit2 = 16
        self.unit3 = 16
        self.model = self.Model()

    def ReadData(self):
        """
            1. Read Data
        """
        dataSet = pd.read_excel(self.filePath)
        return dataSet

    def DrawHeatMap(self):
        """
            2. Drawing Heat Maps
        """
        result = self.dataSet.corr()
        plt.figure(figsize=(20, 20), dpi=100)
        sns.heatmap(result, annot=True, vmax=1, square=True)
        plt.savefig(self.savePicturePath)
        plt.title("Heat Map", fontsize='xx-large', fontweight='heavy')
        plt.show()

    def GetCorrelationCoefficient(self):
        """
            3. Acquisition of Features Related to Work Status
        """
        result = self.dataSet.corr()
        resultData = pd.DataFrame(data=result["Label"], index=result.index)
        resultData.drop(["Label"], axis=0, inplace=True)
        resultData = abs(resultData)
        resultData = resultData[resultData["Label"] >= 0]
        return resultData.index

    def Standard(self):
        """
            4. Data Standardisation
        """
        transform_Standard = StandardScaler()
        data_standard = transform_Standard.fit_transform(self.newFeature)
        return data_standard

    def SplitTrainAndTest(self):
        """
            5. Division of Training and Test Datasets
        """
        target = np.array(tf.one_hot(self.target,self.classificationNumber))
        newFeatures = self.data_standard
        x_train, x_test, y_train, y_test = train_test_split(newFeatures,target, train_size=0.7, shuffle=True)
        return x_train, x_test, y_train, y_test

    def Model(self):
        """
            6. Core of the Model
        """
        w = RandomNormal(mean=0.0, stddev=0.05, seed=None)
        b = Zeros()
        model = Sequential()
        model.add(Dense(units=self.unit1, input_dim=self.x_train.shape[1], kernel_initializer=w, bias_initializer=b,activation=relu))
        model.add(BatchNormalization())
        model.add(Dense(units=self.unit2, kernel_initializer=w, bias_initializer=b, activation=relu))
        model.add(BatchNormalization())
        model.add(Dense(units=self.unit3, kernel_initializer=w, bias_initializer=b, activation=relu))
        model.add(BatchNormalization())
        model.add(Dense(units=self.classificationNumber, activation=softmax))
        return model

    def DrawResult(self,history):
        """
            7. Drawing Training Error
        """
        plt.figure(figsize=(20, 8), dpi=100)
        plt.plot(range(len(history.history['loss'])), history.history['loss'], lw=4, color="blue", label='Training Dataset Error')
        plt.plot(range(len(history.history['val_loss'])), history.history['val_loss'], lw=4, color="red", label='Validation Dataset Error')
        plt.legend(loc='upper left', frameon=True)
        plt.grid(linestyle='-.')
        plt.title("Training Dataset Error and Validation Dataset Error Plotting Results")
        plt.show()

        plt.figure(figsize=(20, 8), dpi=100)
        plt.plot(range(len(history.history['accuracy'])), history.history['accuracy'], lw=4, color="red", label='Training Dataset Accuracy')
        plt.legend(loc='upper left', frameon=True)
        plt.grid(linestyle='-.')
        plt.title("Training Dataset Accuracy Plotting Result")
        plt.show()

    def DrawConfusionMatrix(self,y_true,y_pred):
        """
            8. Drawing Confusion Matrix
        """
        # 生成labels
        labels = []
        for i in range(self.classificationNumber):
            labels.insert(i,i+1)

        y_true = np.argmax(y_true, axis=1) + 1
        y_pred = np.argmax(y_pred,axis=1) + 1
        C = confusion_matrix(y_true, y_pred, labels=labels)
        plt.matshow(C, cmap=plt.cm.Greens)
        for i in range(len(C)):
            for j in range(len(C)):
                plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
        plt.ylabel('Real-Value')
        plt.xlabel('Predicted-Value')
        plt.title("DNN-Confusion Matrix")
        plt.show()
    
    def FaultDiagnosis(self,test_path,saveModelFile,file_path):
        """
            9. Fault Diagnosis
        """
        # Read New Fault Data
        test_data = pd.read_excel(io=test_path)
        # Make Feature Dataset X and Tag Dataset Y
        Y = test_data["Label"]
        target = np.array(tf.one_hot(Y,self.classificationNumber))
        test_data.drop(["Label"], axis=1, inplace=True)
        X = np.array(test_data)
        # Data Standardisation
        transform_Standard = StandardScaler()
        data_standard = transform_Standard.fit_transform(X)
        # Fault Diagnosis
        model = models.load_model(saveModelFile)
        y_pred = np.around(model.predict(data_standard),decimals=0)
        score = accuracy_score(y_true=target, y_pred=y_pred)
        print("Model Diagnostic Accuracy：",'%.2f%%' % (score * 100))
        # Output Diagnostic Results
        pred = []
        for i in range(y_pred.shape[0]):
            pred.insert(i,np.argmax(y_pred[i,:]) + 1)
        real = []
        for i in range(y_pred.shape[0]):
            real.insert(i,np.argmax(target[i,:]) + 1)
        result = np.concatenate((np.array(real).reshape(y_pred.shape[0],1),
                                 np.array(pred).reshape(y_pred.shape[0],1)),axis=1)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                if result[i,j] == 1:
                    result[i,j] = 14
                else:
                    result[i,j] = result[i,j] - 1
        result = pd.DataFrame(data=result,columns=["True","Pred"])
        result.to_excel(file_path,index=False)
     
if __name__ == '__main__':
    # Start Training the Model
    test_number = 1;train_score = 0;test_score = 0;model_score = 0
    for i in range(test_number):
        filePath = "E:\博士资料\本地论文\自己写的论文\第三篇论文\论文\数据\训练数据\FaultDataset.xlsx"
        savePicturePath = "E:\博士资料\本地论文\自己写的论文\第三篇论文\论文\图片\PNG\DNN热力图.png"
        model = MLP(filePath=filePath, savePicturePath=savePicturePath)
        model.DrawHeatMap()
        saveModelFile = "E:\博士资料\本地论文\自己写的论文\第三篇论文\论文\故障诊断模型\DNN模型\DNN_model.h5"
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(saveModelFile, monitor='val_accuracy', verbose=1,
                                                   save_best_only=True,save_weights_only=False,mode='max')
        ]
        model.model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics='accuracy')
        history = model.model.fit(model.x_train, model.y_train, epochs=500, batch_size=128, shuffle=True, verbose=True
                                  ,validation_split=0.2,validation_freq=3,callbacks=callbacks)
        # Plotting the Training Dataset Error
        model.DrawResult(history=history)
        # Loading Model
        model1 = models.load_model(saveModelFile)
        # Training Dataset Evaluation
        y_train_pred = np.around(model1.predict(model.x_train), decimals=0)
        score = accuracy_score(y_true=model.y_train, y_pred=y_train_pred)
        train_score = train_score + score
        # Test Dataset Evaluation
        y_test_pred = np.around(model1.predict(model.x_test), decimals=0)
        score = accuracy_score(y_true=model.y_test, y_pred=y_test_pred)
        # model.DrawConfusionMatrix(y_test_pred)
        test_score = test_score + score

    print("Classification Accuracy on the Training Dataset:", '%.2f%%' % (train_score * 100/test_number))
    print("Classification Accuracy on the Test Dataset:", '%.2f%%' % (test_score * 100/test_number))

    # Read New Faulty Condition Dataset for Fault Diagnosis
    test_path = "E:\博士资料\本地论文\自己写的论文\第三篇论文\论文\数据\测试数据\NewFaultDataset.xlsx"
    file_path = "E:\博士资料\本地论文\自己写的论文\第三篇论文\论文\数据\诊断结果\DNNResult.xlsx"
    model.FaultDiagnosis(test_path,saveModelFile,file_path)
    

    
    

    



   

    


    
    