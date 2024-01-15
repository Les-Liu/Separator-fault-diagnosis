"""
    Fault Diagnosis of Separator Systems Done Using RESNET Model
"""
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from tensorflow.keras import layers,models,optimizers,initializers
from tensorflow.keras.layers import Conv1D,BatchNormalization,\
    ZeroPadding1D,Input,MaxPooling1D,Flatten,Dense
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

class ResNetModel():

    def __init__(self,filePath,classificationNumber):
        self.filePath = filePath
        self.classificationNumber = classificationNumber
        self.dataSet = self.LoadData()
        self.rows = self.dataSet.shape[0]
        self.columns = self.dataSet.shape[1]
        self.features,self.target = self.ExtendedDimension()
        self.target_OneHot = self.OneHot()
        self.x_train, self.x_test, self.y_train, self.y_test = self.SplitTrainAndTest()
        self.input,self.output = self.CreateModel()

    def LoadData(self):
        """
            1. Read Data
        """
        dataSet = pd.read_excel(self.filePath)
        return dataSet

    def ExtendedDimension(self):
        """
            2. Dimension Expansion
        """
        features = np.expand_dims(self.dataSet.values[:, 0:int(self.columns-1)].astype(float), axis=2)
        target = self.dataSet.values[:, int(self.columns-1)]
        return features,target

    def OneHot(self):
        """
            3. One-hot Coding of Target Values
        """
        encoder = LabelEncoder()
        target_encoded = encoder.fit_transform(self.target)
        target_OneHot = np_utils.to_categorical(target_encoded)
        return target_OneHot

    def SplitTrainAndTest(self):
        """
            4. Division of Training and Test Datasets
        """
        x_train, x_test, y_train, y_test = train_test_split(
            self.features,
            self.target_OneHot,
            train_size=0.7,
            shuffle=True,
            random_state=64)
        return x_train, x_test, y_train, y_test

    def BasicBlock(self,x, nb_filter, kernel_size, strides=1, padding='same', name=None):
        """
            5. Building the Foundation Module
        """
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None

        x = Conv1D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
        x = BatchNormalization(axis=1, name=bn_name)(x)
        return x

    def ResBlock(self, input, nb_filter, kernel_size, strides=1, with_conv_shortcut=False):
        """
            6. Build the Residual Module
        """
        x = self.BasicBlock(input, nb_filter=nb_filter, kernel_size=1, strides=strides, padding='same')
        x = self.BasicBlock(x, nb_filter=nb_filter, kernel_size=3, padding='same')
        x = self.BasicBlock(x, nb_filter=nb_filter, kernel_size=1, padding='same')

        if with_conv_shortcut:
            shortcut = self.BasicBlock(input, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
            x = layers.add([x, shortcut])
            return x
        else:
            x = layers.add([x, input])
            return x

    def CreateModel(self):
        """
            7. Core of the Model
        """
        # w = initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
        # b = initializers.Zeros()
        input = Input(shape=(int(self.columns-1), 1))
        x = ZeroPadding1D(3)(input)
        x = self.BasicBlock(x, nb_filter=32, kernel_size=4, strides=2, padding='same')
        x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
        x = self.ResBlock(x, nb_filter=32, kernel_size=4, strides=2, with_conv_shortcut=True)
        x = self.ResBlock(x, nb_filter=32, kernel_size=4)
        x = MaxPooling1D(pool_size=3,padding="same")(x)
        x = Flatten()(x)
        x = BatchNormalization()(x)
        output = Dense(self.classificationNumber, activation=tf.nn.softmax)(x)
        return input,output

    def DrawResult(self,history):
        """
            8. Drawing Training Error
        """
        plt.figure(figsize=(20, 8), dpi=100)
        plt.plot(range(len(history.history['loss'])), history.history['loss'], lw=4, color="blue", label='Training Dataset Error')
        plt.legend(loc='upper left', frameon=True)
        plt.grid(linestyle='-.')
        plt.title("Training Dataset Error Plotting Result")
        plt.show()

        plt.figure(figsize=(20, 8), dpi=100)
        plt.plot(range(len(history.history['accuracy'])), history.history['accuracy'], lw=4, color="red", label='Training Dataset Accuracy')
        plt.legend(loc='upper left', frameon=True)
        plt.grid(linestyle='-.')
        plt.title("Training Dataset Accuracy Plotting Result")
        plt.show()

    def DrawConfusionMatrix(self,y_pred,y_true):
        """
            9. Drawing Confusion Matrix
        """
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
        plt.title("RESNET-Confusion Matrix")
        plt.show()
        return C
    
    def FaultDiagnosis(self,test_path,saveModelFile,file_path):
        """
            10. Fault Diagnosis
        """
        # Read New Fault Data
        test_data = pd.read_excel(io=test_path)
        # Make Feature Dataset X and Tag Dataset Y
        features = np.expand_dims(test_data.values[:, 0:int(test_data.shape[1]-1)].astype(float), axis=2)
        target = test_data.values[:, int(test_data.shape[1]-1)]
        encoder = LabelEncoder()
        target_encoded = encoder.fit_transform(target)
        target_OneHot = np_utils.to_categorical(target_encoded)
        X = np.array(features)
        # Fault Diagnosis
        model = models.load_model(saveModelFile)
        y_pred = np.around(model.predict(X),decimals=0)
        score = accuracy_score(y_true=target_OneHot, y_pred=y_pred)
        print("Model Diagnostic Accuracy:",'%.2f%%' % (score * 100))
        # Output Diagnostic Results
        pred = []
        for i in range(y_pred.shape[0]):
            pred.insert(i,np.argmax(y_pred[i,:]) + 1)
        real = []
        for i in range(y_pred.shape[0]):
            real.insert(i,np.argmax(target_OneHot[i,:]) + 1)
        result = np.concatenate((np.array(real).reshape(y_pred.shape[0],1),
                                 np.array(pred).reshape(y_pred.shape[0],1)),axis=1)
        # for i in range(result.shape[0]):
        #     for j in range(result.shape[1]):
        #         if result[i,j] == 1:
        #             result[i,j] = 14
        #         else:
        #             result[i,j] = result[i,j] - 1
        result = pd.DataFrame(data=result,columns=["True","Pred"])
        result.to_excel(file_path,index=False)

if __name__ == '__main__':
    # Start Training the Model
    test_number = 1;train_score = 0;test_score = 0;model_score = 0
    for i in range(test_number):
        filePath = "E:\博士资料\本地论文\自己写的论文\第三篇论文\论文\数据\训练数据\FaultDataset.xlsx"
        classificationNumber = 14
        resnet = ResNetModel(filePath=filePath,classificationNumber=classificationNumber)
        model = models.Model(inputs=resnet.input,outputs=resnet.output)
        optimizer = optimizers.Adam(learning_rate=0.001)
        saveModelFile = "E:/博士资料/本地论文/自己写的论文/第三篇论文/论文/故障诊断模型/ResNet模型/ResNet_model.h5"
        callbacks = [
        # tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=10,min_delta=0.001,mode='max'),
        tf.keras.callbacks.ModelCheckpoint(saveModelFile, monitor='val_accuracy', verbose=1,
                                                   save_best_only=True,save_weights_only=False,mode='max')
        ]
        # Network Training
        model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        history = model.fit(resnet.x_train, resnet.y_train, epochs=500, shuffle=True, verbose=True
                    , validation_split=0.2, validation_freq=3, batch_size=128,callbacks=callbacks)
        resnet.DrawResult(history=history)
        # Load Model
        model = models.load_model(saveModelFile)
        # Training Dataset Evaluation
        y_train_pred = np.around(model.predict(resnet.x_train), decimals=0)
        score = accuracy_score(y_true=resnet.y_train, y_pred=y_train_pred)
        train_score = train_score + score
        # Test Dataset Evaluation
        y_test_pred = np.around(model.predict(resnet.x_test), decimals=0)
        score = accuracy_score(y_true=resnet.y_test, y_pred=y_test_pred)
        test_score = test_score + score
        # resnet.DrawConfusionMatrix(y_test_pred,resnet.y_test)

    print("Classification Accuracy on the Training Dataset:", '%.2f%%' % (train_score * 100/test_number))
    print("Classification Accuracy on the test Dataset:", '%.2f%%' % (test_score * 100/test_number))

    # 读取新的故障工况数据集，进行故障诊断
    test_path = "E:\博士资料\本地论文\自己写的论文\第三篇论文\论文\数据\测试数据\NewFaultDataset.xlsx"
    file_path = "E:\博士资料\本地论文\自己写的论文\第三篇论文\论文\数据\诊断结果\RESNETResult.xlsx"
    resnet.FaultDiagnosis(test_path,saveModelFile,file_path)
