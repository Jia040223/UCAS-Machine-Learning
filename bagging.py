import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from scipy.stats import mode
from openpyxl import load_workbook

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
        self.visual_train_data = {}
        self.audio_train_data = {}
        #map,所有的key值对应的训练数据均在其中
        self.text_test_data = {}
        self.visual_test_data = {}
        self.audio_test_data = {}
        #map,所有的key值对应的测试数据均在其中

        self.text_X_train = []
        self.visual_X_train = []
        self.audio_X_train = []
        self.Y_train = []
        #这里整合之后的数据并不是用来训练的，而是用来让基学习器算权重的
        
        self.text_X_test = []
        self.visual_X_test = []
        self.audio_X_test = []

    def load_train_data(self):
        for key,value in self.audio_features.items():
            if key in self.train_ids:
                self.audio_train_data[key] = value
        # df_load_audio_train = pd.DataFrame(list(self.audio_train_data.items()))
        # df_load_audio_train.to_csv('load_audio_train.csv', index=False)  
        # print("finish load_audio_train to csv")     
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
        # df_load_audio_train = pd.DataFrame(list(self.train_label.items()))
        # df_load_audio_train.to_csv('load_audio_train.csv', index=False)  
        # print("finish train_label to csv") 
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
            # # print(list.shape)
            # df_load_audio_train = pd.DataFrame(list)
            # df_load_audio_train.to_csv('load_audio_train.csv', index=False)  
            # print("finish train_label to csv") 
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
        # df_load_audio_train = pd.DataFrame(self.text_X_train)
        # df_load_audio_train.to_csv('load_audio_train.csv', index=False)  
        # print("finish train_label to csv") 
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
    def __init__(self, train_data, train_label, test_data,train_weight_data):
        # self.model = LogisticRegression(max_iter=10000)
        self.model = RandomForestClassifier()
        # self.model = DecisionTreeClassifier()
        # self.model=LogisticRegression(solver='liblinear')
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.train_weight_data = train_weight_data
        self.Y_predict = np.array([])

    def weight_predict(self):
        self.Y_predict = self.model.predict(self.train_weight_data)
        return self.Y_predict
       
    def predict(self):
        df_ = pd.DataFrame(self.train_data)
        df_.to_csv('check_self_train_data.csv', index=False) 
        self.model.fit(self.train_data, self.train_label)
        self.Y_predict = self.model.predict(self.test_data)
        return self.Y_predict
        # print(Y_predict)

    def save(self):
        Y_predict = self.Y_predict.astype(int)
        Y_predict = Y_predict.reshape(-1,1)
        np.savetxt('dicision_output.csv', Y_predict, delimiter = ',', fmt = '%d')

    def cal_correct_rate(self):
        Y_true = np.loadtxt('C:/Users/32175/Desktop/Machine_Learning/UCAS-Machine-Learning/reference_answer.csv', dtype = int)
        weighted_f1 = f1_score(Y_true, self.Y_predict, average='weighted')
        print('weighted_f1:',weighted_f1)


if __name__ == '__main__':
    Y_true = np.loadtxt('C:/Users/32175/Desktop/Machine_Learning/UCAS-Machine-Learning/reference_answer.csv', dtype = int)

    text_DecisionTreemodles = {}
    visual_DecisionTreemodles = {}
    audio_DecisionTreemodles = {}

    text_X_train = []
    visual_X_train = []
    audio_X_train = []

    Y_train = []
    text_Y_predict = {}
    visual_Y_predict = {}
    audio_Y_predict = {}

    train_weight_text_predict = {}
    train_weight_visual_predict = {}
    train_weight_audio_predict = {}
    #在训练集样本上进行预测
    train_weight_text_count = {}
    train_weight_visual_count = {}
    train_weight_audio_count = {}
    #在训练集样本上预测的结果
    general_text_count = 0
    general_visual_count = 0
    general_audio_count = 0

    pred_text   = np.arange(0,len(Y_true),1)
    pred_visual = np.arange(0,len(Y_true),1)
    pred_audio  = np.arange(0,len(Y_true),1)


    # random_text_Y_predict = {}
    data_loader = DataLoader('C:/Users/32175/Desktop/Machine_Learning/UCAS-Machine-Learning/DataSet')
    data_loader.load_train_data()
    data_loader.load_train_label()
    data_loader.load_test_data()#加载完所有的训练数据，标签和样本
    data_loader.DataProcess() 

    print(data_loader.text_train_data)
    for key,list in data_loader.text_train_data.items():
        text_X_train = list
        Y_train = data_loader.train_label[key]

        random_keys = random.sample(data_loader.text_train_data.keys(), 1)
        for array in data_loader.text_train_data[random_keys[0]]:
            # print(data_loader.text_train_data[random_keys[0]])
            text_X_train.append(array)
        for array in data_loader.train_label[random_keys[0]]:
            # print(data_loader.train_label[random_keys[0]])
            Y_train.append(array)
        # for array in data_loader.text_train_data[random_keys[1]]:
        #     # print(data_loader.text_train_data[random_keys[1]])
        #     text_X_train.append(array)
        # for array in data_loader.train_label[random_keys[1]]:
        #     # print(data_loader.train_label[random_keys[1]])
        #     Y_train.append(array)
        # for array in data_loader.text_train_data[random_keys[2]]:
        #     # print(data_loader.text_train_data[random_keys[2]])
        #     text_X_train.append(array)
        # for array in data_loader.train_label[random_keys[2]]:
        #     # print(data_loader.train_label[random_keys[2]])
        #     Y_train.append(array)
        #将dataloader中的text训练数据和对应的label给印出来
        print("end ",key)
        text_X_train = np.array(text_X_train)
        Y_train = np.array(Y_train)
        #用np包进行转换
        #这里实现了每一个key值对应的text和对应的label都训练出来一个model然后进行预测
        text_DecisionTreemodles[key] = DecisionTree(text_X_train, Y_train, data_loader.text_X_test, data_loader.text_X_train)
        #初始化一个model
        text_Y_predict[key] = text_DecisionTreemodles[key].predict()
        #根据该模型进行预测
        train_weight_text_predict[key] = text_DecisionTreemodles[key].weight_predict()
        #训练样本上的测试
        train_weight_text_count[key] = 0
        for i in range(len(data_loader.Y_train)):#查看训练集上的标签
            if data_loader.Y_train[i] == train_weight_text_predict[key][i] :
                train_weight_text_count[key] += 1
        general_text_count += train_weight_text_count[key]
        # pred_final = mode(np.c_[pred_text, pred_audio, pred_visual], axis=1)[0]
    
    df_text = pd.DataFrame(text_Y_predict)
    df_text.to_csv('check_text_Y_predict.csv', index=False) 

    df_weight_text = pd.DataFrame(train_weight_text_predict)
    df_weight_text.to_csv('check_weight_text_Y_predict.csv', index=False)
    #查看在训练集上的结果 
    #打印出不同的基学习器对于text特征的预测
    #先训练出151个基学习器，然后根据他们在训练集上的误差来分配预测的权重，最终实现bagging加权平均法
    # for key in text_DecisionTreemodles.items():
    #     train_weight_text_predict[key] = text_DecisionTreemodles[key].weight_predict()
    #     #训练样本上的测试
    #     train_weight_text_count[key] = 0
    #     for i in range(len(data_loader.Y_train)):#查看训练集上的标签
    #         if data_loader.Y_train[i] == train_weight_text_predict[key][i] :
    #             train_weight_text_count[key] += 1
    #     general_text_count += train_weight_text_count[key]

    print(train_weight_text_count)
    df = pd.DataFrame(train_weight_text_count,index=[0])
    df.to_csv('train_weight_text_count.csv') 

    for i in range(len(Y_true)):
        # print(len(Y_true))
        # print(i)
        pred_text[i] = 0
        temp_pred_0 = 0
        temp_pred_1 = 0
        temp_pred_2 = 0
        temp_pred_3 = 0
        temp_pred_4 = 0
        temp_pred_5 = 0
        # print(train_weight_text_count.items())
        for key in train_weight_text_count:
            if text_Y_predict[key][i] == 0 :
                temp_pred_0 += train_weight_text_count[key]
            elif text_Y_predict[key][i] == 1 :
                temp_pred_1 += train_weight_text_count[key]
            elif text_Y_predict[key][i] == 2 :
                temp_pred_2 += train_weight_text_count[key]
            elif text_Y_predict[key][i] == 3 :
                temp_pred_3 += train_weight_text_count[key]
            elif text_Y_predict[key][i] == 4 :
                temp_pred_4 += train_weight_text_count[key]
            elif text_Y_predict[key][i] == 5 :
                temp_pred_5 += train_weight_text_count[key]
        if i == 0: 
            print(temp_pred_0)
            print(temp_pred_1)
            print(temp_pred_2)
            print(temp_pred_3)
            print(temp_pred_4)
            print(temp_pred_5)
        judge = max(temp_pred_0,temp_pred_1,temp_pred_2,temp_pred_3,temp_pred_4,temp_pred_5)
        if judge == temp_pred_0:
            pred_text[i] = 0
        elif judge == temp_pred_1:
            pred_text[i] = 1
        elif judge == temp_pred_2:
            pred_text[i] = 2
        elif judge == temp_pred_3:
            pred_text[i] = 3
        elif judge == temp_pred_4:
            pred_text[i] = 4
        elif judge == temp_pred_5:
            pred_text[i] = 5
        if i == 0:
            print(pred_text[i])

    # pred_text = df_text.mode(axis='columns')#根据众数进行选举
    df_text_bagging = pd.DataFrame(pred_text)
    df_text_bagging.to_csv('bagging_text.csv', index=False) 

    # #打印出不同的基学习器对于text特征的预测

    for key,list in data_loader.visual_train_data.items():
        visual_X_train = list
        Y_train = data_loader.train_label[key]
        df = pd.DataFrame(visual_X_train)
        df.to_csv('check_visual_X.csv', index=False) 
        df = pd.DataFrame(Y_train)
        df.to_csv('check_Y_train.csv', index=False) 
        #将dataloader中的visual训练数据和对应的label给印出来
        visual_X_train = np.array(visual_X_train)
        Y_train = np.array(Y_train)
        #用np包进行转换
        #这里实现了每一个key值对应的visual和对应的label都训练出来一个model然后进行预测
        visual_DecisionTreemodles[key] = DecisionTree(visual_X_train, Y_train, data_loader.visual_X_test, data_loader.visual_X_train)
        #初始化一个model
        visual_Y_predict[key] = visual_DecisionTreemodles[key].predict()
        #根据该模型进行预测
        train_weight_visual_predict[key] = visual_DecisionTreemodles[key].weight_predict()
        #训练样本上的测试
        train_weight_visual_count[key] = 0
        for i in range(len(data_loader.Y_train)):#查看训练集上的标签
            if data_loader.Y_train[i] == train_weight_visual_predict[key][i] :
                train_weight_visual_count[key] += 1
        general_visual_count += train_weight_visual_count[key]
        # pred_final = mode(np.c_[pred_text, pred_audio, pred_visual], axis=1)[0]

    # df = pd.DataFrame(list(visual_Y_predict.items))
    df_visual = pd.DataFrame(visual_Y_predict)
    df_visual.to_csv('check_visual_Y_predict.csv', index=False) 
    #打印出不同的基学习器对于visual特征的预测

    df_weight_visual = pd.DataFrame(train_weight_visual_predict)
    df_weight_visual.to_csv('check_weight_visual_Y_predict.csv', index=False)
    #查看在训练集上的结果 
    #打印出不同的基学习器对于visual特征的预测
    #先训练出151个基学习器，然后根据他们在训练集上的误差来分配预测的权重，最终实现bagging加权平均法

    print(train_weight_visual_count)
    df = pd.DataFrame(train_weight_visual_count,index=[0])
    df.to_csv('train_weight_visual_count.csv') 

    for i in range(len(Y_true)):
        # print(len(Y_true))
        # print(i)
        pred_visual[i] = 0
        temp_pred_0 = 0
        temp_pred_1 = 0
        temp_pred_2 = 0
        temp_pred_3 = 0
        temp_pred_4 = 0
        temp_pred_5 = 0
        # print(train_weight_visual_count.items())
        for key in train_weight_visual_count:
            if visual_Y_predict[key][i] == 0 :
                temp_pred_0 += train_weight_visual_count[key]
            elif visual_Y_predict[key][i] == 1 :
                temp_pred_1 += train_weight_visual_count[key]
            elif visual_Y_predict[key][i] == 2 :
                temp_pred_2 += train_weight_visual_count[key]
            elif visual_Y_predict[key][i] == 3 :
                temp_pred_3 += train_weight_visual_count[key]
            elif visual_Y_predict[key][i] == 4 :
                temp_pred_4 += train_weight_visual_count[key]
            elif visual_Y_predict[key][i] == 5 :
                temp_pred_5 += train_weight_visual_count[key]
        if i == 0:
            print(temp_pred_0)
            print(temp_pred_1)
            print(temp_pred_2)
            print(temp_pred_3)
            print(temp_pred_4)
            print(temp_pred_5)
        judge = max(temp_pred_0,temp_pred_1,temp_pred_2,temp_pred_3,temp_pred_4,temp_pred_5)
        if judge == temp_pred_0:
            pred_visual[i] = 0
        elif judge == temp_pred_1:
            pred_visual[i] = 1
        elif judge == temp_pred_2:
            pred_visual[i] = 2
        elif judge == temp_pred_3:
            pred_visual[i] = 3
        elif judge == temp_pred_4:
            pred_visual[i] = 4
        elif judge == temp_pred_5:
            pred_visual[i] = 5
        if i == 0:
            print(pred_visual[i])


    # print(visual_Y_predict.values())
    df_visual_bagging = pd.DataFrame(pred_visual)
    df_visual_bagging.to_csv('bagging_visual.csv', index=False) 

    for key,list in data_loader.audio_train_data.items():
        audio_X_train = list
        Y_train = data_loader.train_label[key]
        #将dataloader中的audio训练数据和对应的label给印出来
        audio_X_train = np.array(audio_X_train)
        Y_train = np.array(Y_train)
        #用np包进行转换
        #这里实现了每一个key值对应的audio和对应的label都训练出来一个model然后进行预测
        audio_DecisionTreemodles[key] = DecisionTree(audio_X_train, Y_train, data_loader.audio_X_test, data_loader.audio_X_train)
        #初始化一个model
        audio_Y_predict[key] = audio_DecisionTreemodles[key].predict()
        #根据该模型进行预测
        train_weight_audio_predict[key] = audio_DecisionTreemodles[key].weight_predict()
        #训练样本上的测试
        train_weight_audio_count[key] = 0
        for i in range(len(data_loader.Y_train)):#查看训练集上的标签
            if data_loader.Y_train[i] == train_weight_audio_predict[key][i] :
                train_weight_audio_count[key] += 1
        general_audio_count += train_weight_audio_count[key]
        # pred_final = mode(np.c_[pred_text, pred_audio, pred_visual], axis=1)[0]

    df_audio = pd.DataFrame(audio_Y_predict)
    df_audio.to_csv('check_audio_Y_predict.csv', index=False) 
    #打印出不同的基学习器对于audio特征的预测

    df_weight_audio = pd.DataFrame(train_weight_audio_predict)
    df_weight_audio.to_csv('check_weight_audio_Y_predict.csv', index=False)
    #查看在训练集上的结果 
    #打印出不同的基学习器对于audio特征的预测
    #先训练出151个基学习器，然后根据他们在训练集上的误差来分配预测的权重，最终实现bagging加权平均法

    print(train_weight_audio_count)
    df = pd.DataFrame(train_weight_audio_count,index=[0])
    df.to_csv('train_weight_audio_count.csv') 

    for i in range(len(Y_true)):
        # print(len(Y_true))
        # print(i)
        pred_audio[i] = 0
        temp_pred_0 = 0
        temp_pred_1 = 0
        temp_pred_2 = 0
        temp_pred_3 = 0
        temp_pred_4 = 0
        temp_pred_5 = 0
        # print(train_weight_audio_count.items())
        for key in train_weight_audio_count:
            if audio_Y_predict[key][i] == 0 :
                temp_pred_0 += train_weight_audio_count[key]
            elif audio_Y_predict[key][i] == 1 :
                temp_pred_1 += train_weight_audio_count[key]
            elif audio_Y_predict[key][i] == 2 :
                temp_pred_2 += train_weight_audio_count[key]
            elif audio_Y_predict[key][i] == 3 :
                temp_pred_3 += train_weight_audio_count[key]
            elif audio_Y_predict[key][i] == 4 :
                temp_pred_4 += train_weight_audio_count[key]
            elif audio_Y_predict[key][i] == 5 :
                temp_pred_5 += train_weight_audio_count[key]
            
        if i == 0:
            print(temp_pred_0)
            print(temp_pred_1)
            print(temp_pred_2)
            print(temp_pred_3)
            print(temp_pred_4)
            print(temp_pred_5)
        judge = max(temp_pred_0,temp_pred_1,temp_pred_2,temp_pred_3,temp_pred_4,temp_pred_5)
        if judge == temp_pred_0:
            pred_audio[i] = 0
        elif judge == temp_pred_1:
            pred_audio[i] = 1
        elif judge == temp_pred_2:
            pred_audio[i] = 2
        elif judge == temp_pred_3:
            pred_audio[i] = 3
        elif judge == temp_pred_4:
            pred_audio[i] = 4
        elif judge == temp_pred_5:
            pred_audio[i] = 5
        if i == 0:
            print(pred_audio[i])

    # print(audio_Y_predict.values())
    df_audio_bagging = pd.DataFrame(pred_audio)
    df_audio_bagging.to_csv('bagging_audio.csv', index=False) 


    # weighted_f2 = f1_score(Y_true, pred_text[0], average='weighted')
    # weighted_f3 = f1_score(Y_true, pred_visual[0], average='weighted')
    # weighted_f4 = f1_score(Y_true, pred_audio[0], average='weighted')
    #计算f1_score
    count_2 = 0
    count_3 = 0
    count_4 = 0
    for i in range(len(Y_true)):
        if Y_true[i] == pred_text[i] :
            count_2 += 1
    for i in range(len(Y_true)):
        if Y_true[i] == pred_visual[i] :
            count_3 += 1
    for i in range(len(Y_true)):
        if Y_true[i] == pred_audio[i] :
            count_4 += 1
    #计算正确率

    # print('weighted_f1:',weighted_f1)   
    # print('weighted_f2:',weighted_f2)  
    # print('weighted_f3:',weighted_f3)  
    # print('weighted_f4:',weighted_f4)
    print('rate_f2:',count_2/len(Y_true)) 
    print('rate_f3:',count_3/len(Y_true)) 
    print('rate_f4:',count_4/len(Y_true)) 
