import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from scipy.stats import mode

class DataLoader(object):
    def __init__(self, DataPath):
        self.train_ids = set(pd.read_csv(DataPath+'/train_ids.csv')['ids'])
        self.test_ids = set(pd.read_csv(DataPath+'/test_ids.csv')['ids'])
        self.audio_features = pd.read_pickle(DataPath+'/audio_features.pkl')
        self.text_features = pd.read_pickle(DataPath+'/text_features.pkl')
        self.visual_features = pd.read_pickle(DataPath+'/visual_features.pkl')
        self.all_label = pd.read_pickle(DataPath+'/train_label.pkl')
        self.dimension = 1024
        self.train_label = {}
        self.test_data = {}
        self.train_data = {}
        self.text_train_data = {}
        self.text_test_data = {}
        self.visual_train_data = {}
        self.visual_test_data = {}
        self.audio_train_data = {}
        self.audio_test_data = {}
        self.text_X_train = []
        self.visual_X_train = []
        self.audio_X_train = []
        self.text_Y_train = []
        self.visual_Y_train = []
        self.audio_Y_train = []
        self.Y_train = []
        self.text_X_test = []
        self.visual_X_test = []
        self.audio_X_test = []

    def load_train_data(self):
        for key,value in self.audio_features.items():
            if key in self.train_ids:
                self.audio_train_data[key] = value
        for key,value in self.text_features.items():
            if key in self.train_ids:
                self.text_train_data[key] = value
        for key,value in self.visual_features.items():
            if key in self.train_ids:
                self.visual_train_data[key] = value
        # print(self.visual_train_data)
        # for key,list in self.train_data.items():
        #     for array in list:
        #         print(len(array))
        #         self.dimension = max(self.dimension,len(array))
        # return self.train_data

    def load_train_label(self):
        for key,value in self.all_label.items():
            if key in self.train_ids:
                self.train_label[key] = value
        # return self.train_label

    def load_test_data(self):
        for key,value in self.audio_features.items():
            if key in self.test_ids:
                self.audio_test_data[key] = value
        for key,value in self.text_features.items():
            if key in self.test_ids:
                self.text_test_data[key] = value
        for key,value in self.visual_features.items():
            if key in self.test_ids:
                self.visual_test_data[key] = value
        # return self.test_data
    
    def DataProcess(self):
        for key,list in self.text_train_data.items():
            for array in list:
                self.text_X_train.append(array)
        for key,list in self.visual_train_data.items():
            for array in list:
                self.visual_X_train.append(array)
        for key,list in self.audio_train_data.items():
            for array in list:
                self.audio_X_train.append(array)
        for key,list in self.train_label.items():
            for array in list:
                self.Y_train.append(array)
        for key,list in self.text_test_data.items():
            for array in list:
                self.text_X_test.append(array)
        for key,list in self.visual_test_data.items():
            for array in list:
                self.visual_X_test.append(array)
        for key,list in self.audio_test_data.items():
            for array in list:
                self.audio_X_test.append(array)
        self.text_X_train = np.array(self.text_X_train)
        self.visual_X_train = np.array(self.visual_X_train)
        self.audio_X_train = np.array(self.audio_X_train)
        self.Y_train = np.array(self.Y_train)
        self.text_X_test = np.array(self.text_X_test)
        self.visual_X_test = np.array(self.visual_X_test)
        self.audio_X_test = np.array(self.audio_X_test)
    
    def show_data_info(self):
        print('text_X_train:',self.text_X_train.shape)
        print('visual_X_train:',self.visual_X_train.shape)
        print('audio_X_train:',self.audio_X_train.shape)
        print('Y_train:',self.Y_train.shape)
        print('text_X_test:',self.text_X_test.shape)
        print('visual_X_test:',self.visual_X_test.shape)
        print('audio_X_test:',self.audio_X_test.shape)



class DecisionTree(object):
    def __init__(self, train_data, train_label, test_data):
        self.model = RandomForestClassifier()
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.Y_predict = np.array([])
       
    def predict(self):
        self.model.fit(self.train_data, self.train_label)
        self.Y_predict = self.model.predict(self.test_data)
        return self.Y_predict
        # print(Y_predict)

    def save(self):
        Y_predict = self.Y_predict.astype(int)
        Y_predict = Y_predict.reshape(-1,1)
        np.savetxt('dicision_output.csv', Y_predict, delimiter = ',', fmt = '%d')

    def cal_correct_rate(self):
        Y_true = np.loadtxt('F:/CODES/Python/UCAS-Machine-Learning/reference_answer.csv', dtype = int)
        weighted_f1 = f1_score(Y_true, self.Y_predict, average='weighted')
        print('weighted_f1:',weighted_f1)


if __name__ == '__main__':
    data_loader = DataLoader('C:/Users/32175/Desktop/Machine_Learning/UCAS-Machine-Learning/DataSet')
    data_loader.load_train_data()
    data_loader.load_train_label()
    data_loader.load_test_data()
    data_loader.DataProcess() 
    data_loader.show_data_info()
    text_X_train = data_loader.text_X_train
    visual_X_train = data_loader.visual_X_train
    audio_X_train = data_loader.audio_X_train
    Y_train = data_loader.Y_train

    text_X_test = data_loader.text_X_test
    visual_X_test = data_loader.visual_X_test
    audio_X_test = data_loader.audio_X_test
    text_model = DecisionTree(text_X_train, Y_train, text_X_test)
    pred_text = text_model.predict()
    visual_model = DecisionTree(visual_X_train, Y_train, visual_X_test)
    pred_visual = visual_model.predict()
    audio_model = DecisionTree(audio_X_train, Y_train, audio_X_test)
    pred_audio = audio_model.predict()
    pred_final = mode(np.c_[pred_text, pred_audio, pred_visual], axis=1)[0]

    Y_true = np.loadtxt('C:/Users/32175/Desktop/Machine_Learning/UCAS-Machine-Learning/reference_answer.csv', dtype = int)
    weighted_f1 = f1_score(Y_true, pred_final, average='weighted')
    # count_2 = 0
    # for i in range(len(Y_true)):
    #     if Y_true[i] == pred_final[0][i] :
    #         count_2 += 1
    df = pd.DataFrame(pred_final)
    df.to_csv('main_pred.csv', index=False) 
    # print('rate_f2:',count_2/len(Y_true))
    print('weighted_f1:',weighted_f1)   
