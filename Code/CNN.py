"""
    Fault Diagnosis of Separator Systems Done Using CNN Model
"""
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv1D,Flatten,MaxPooling1D,BatchNormalization
from tensorflow.keras.activations import relu,softmax
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score,confusion_matrix

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

"""
    Building CNN Model
"""
class CNN():
    def __init__(self,filePath):
        self.filePath = filePath # Read File Path
        self.dataSet,self.target = self.ReadData() # Getting the Training Dataset
        self.classificationNumber = 14 # Number of Classifications
        self.data_standard = self.Standard()
        self.x_train, self.x_test, self.y_train, self.y_test = self.SplitTrainAndTest()
        self.model = self.Model()

    def ReadData(self):
        """
            1. Read Data
        """
        dataSet = pd.read_excel(self.filePath)
        Y = dataSet["Label"]
        X = dataSet.drop(["Label"],axis=1)
        return X,Y

    def Standard(self):
        """
            2. Data Standardisation
        """
        transform_Standard = StandardScaler()
        data_standard = transform_Standard.fit_transform(self.dataSet)
        return data_standard

    def SplitTrainAndTest(self):
        """
            3. Division of Training and Test Datasets
        """
        target = np.array(tf.one_hot(self.target,self.classificationNumber))
        x = np.expand_dims(self.data_standard.astype(float), axis=2)
        x_train, x_test, y_train, y_test = train_test_split(x,target, train_size=0.7, shuffle=True)
        return x_train, x_test, y_train, y_test

    def Model(self):
        """
            4. Core of the Model
        """
        model = Sequential()
        model.add(Conv1D(32, 2, input_shape=(self.x_train.shape[1], 1),activation=relu))
        model.add(Conv1D(32, 2, activation= relu))
        model.add(MaxPooling1D(2,padding="same"))
        model.add(Conv1D(32, 2, activation=relu))
        model.add(MaxPooling1D(2,padding="same"))
        model.add(Flatten())
        model.add(BatchNormalization())
        model.add(Dense(self.classificationNumber, activation=softmax))
        return model

    def DrawLoss(self,history):
        """
            5. Drawing Training Error
        """
        plt.figure(figsize=(20, 8), dpi=100)
        plt.plot(range(len(history.history['loss'])), history.history['loss'], lw=4, color="blue", label='Training Dataset Error')
        # plt.plot(range(len(history.history['val_loss'])), history.history['val_loss'], lw=4, color="green", label='验证集误差')
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

    def DrawConfusionMatrix(self,y_true,y_pred):
        """
            6. Drawing Confusion Matrix
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
        plt.title("CNN-Confusion Matrix")
        plt.show()
    
    def FaultDiagnosis(self,test_path,saveModelFile,file_path):
        """
            7. Fault Diagnosis
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
        model = tf.keras.models.load_model(saveModelFile)
        y_pred = np.around(model.predict(data_standard),decimals=0)
        score = accuracy_score(y_true=target, y_pred=y_pred)
        print("Model Diagnostic Accuracy:",'%.2f%%' % (score * 100))
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
        filePath = "F:/博士资料/本地论文/自己写的论文/第三篇论文/论文/Data/Train Data/FaultDataset 50snr.xlsx"
        saveModelFile = "F:/博士资料/本地论文/自己写的论文/第三篇论文/论文/Model/50snr/CNN_model.h5"
        model = CNN(filePath=filePath)
        callbacks = [
            # tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, min_delta=0.001, mode='max'),
            tf.keras.callbacks.ModelCheckpoint(saveModelFile, monitor='val_accuracy', verbose=1,
                                                   save_best_only=True,save_weights_only=False,mode='max')
        ]
        model.model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        history = model.model.fit(model.x_train, model.y_train, epochs=500, batch_size=128, shuffle=True, verbose=True
                                  , validation_split=0.2, validation_freq=3, callbacks=callbacks)
        model.DrawLoss(history=history)
        # Loading Model
        cnn_model = tf.keras.models.load_model(saveModelFile)
        # Training Dataset Evaluation
        y_train_pred = np.around(cnn_model.predict(model.x_train), decimals=0)
        score = accuracy_score(y_true=model.y_train, y_pred=y_train_pred)
        train_score = train_score + score
        # Test Dataset Evaluation
        y_test_pred = np.around(cnn_model.predict(model.x_test), decimals=0)
        score = accuracy_score(y_true=model.y_test, y_pred=y_test_pred)
        test_score = test_score + score
        
    print("Classification Accuracy on the Training Dataset:", '%.2f%%' % (train_score * 100/test_number))
    print("Classification Accuracy on the Test Dataset:", '%.2f%%' % (test_score * 100/test_number))

    # Read New Faulty Condition Dataset for Fault Diagnosis
    test_path = "F:/博士资料/本地论文/自己写的论文/第三篇论文/论文/Data/Test Data/NewFaultDataset.xlsx"
    file_path = "F:/博士资料/本地论文/自己写的论文/第三篇论文/论文/Data/Diagnostic Results/50snr/CNNResult.xlsx"
    model.FaultDiagnosis(test_path,saveModelFile,file_path)
