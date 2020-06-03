#Train.csv文件中，一天中有18项数据，每项数据都有24个不同的观测值
#所以一天中的数据维度为(18,24),总共天数有240天，前9个小时用来训练
#第十个小时用来做lable，故一天中的数据可分为24-10+1=15组，总共的数据
#有240*15=3600组数据，而标签为3600个

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv


#数据预处理
def dataProcess(df):
    x_list, y_list = [], []
    #将df中的空数据填为0
    df = df.replace(['NR'], [0.0])
    #将所有数据转换成float类型
    array = np.array(df).astype(float)
    #将数据集拆分成多个数据帧
    for i in range(0, 4320, 18): #18*240 = 4320（len(array)
        for j in range(15): #24-9=15(组)
            mat = array[i:i+18][9][j:j+9] #截取PM2.5前9项作为训练，i:i+18表示一天中的所有数据项，[9]表示第9行数据，也就是PM2.5，[j:j+9]表示前九小时的数据
            lable = array[i:i+18][9][j+9] #截取PM2.5第十项作为lable，[j+9]表示第十小时的PM2.5的值
            x_list.append(mat)
            y_list.append(lable)
    x = np.array(x_list)
    y = np.array(y_list)
    return x, y, array


#训练模型，从训练集中拿出3200个数据用来训练，400个数据用来验证
def trainModel(x_data, y_data, epoch):
    bias = 0 #初始化偏置
    weight = np.ones(9) #生成一个9列的行向量，并全部初始化为1
    learning_rate = 1 #初始化学习率为1
    reg_rate = 0.001 #初始化正则项系数为0.001
    bias_sum = 0  # 用于存放偏置的梯度平方和
    weight_sum = np.zeros(9)  # 用于存放权重的梯度平方和
    for i in range(epoch):
        b_g = 0 #bias梯度平均值
        w_g = np.zeros(9) #weight梯度平均值
        #在所有数据上计算w和b的梯度
        for j in range(3200):
            b_g += (y_data[j] - weight.dot(x_data[j]) - bias) * (-1) #如果两个一维向量dot，则结果为它们的内积
            for k in range(9):
                w_g[k] += (y_data[j] - weight.dot(x_data[j]) - bias) * (-x_data[j, k])
        #求平均值
        b_g /= 3200
        w_g /= 3200
        for k in range(9):
            w_g[k] += reg_rate * weight[k]
        #adagrad优化方式
        bias_sum += b_g ** 2
        weight_sum += w_g ** 2
        bias -= learning_rate / (bias_sum ** 0.5) * b_g
        weight -= learning_rate / (weight_sum ** 0.5) * w_g

        #每训练200次输出一次误差
        if i % 200 ==0:
            loss = 0
            for j in range(3200):
                loss += (y_data[j] - weight.dot(x_data[j]) - bias) ** 2
            loss /= 3200
            for j in range(9):
                loss += reg_rate * (weight[j] ** 2)
            print('after {} epoch, the loss on the train_set is {:.2f}'.format(i, loss / 2))
    return weight, bias


#验证模型，返回loss
def validateModel(x_val, y_val, weight, bais):
    loss = 0
    for i in range(400):
        loss += (y_val[i] - weight.dot(x_val[i]) - bais) ** 2
    return loss / 400

#绘制PM2.5的图像
def drawPM(x_data, y_data, weight, bias):
    y = weight.dot(x_data[0, 9]) + bias
    plt.plot(x_data[0, 9], y)


#测试数据预处理
def testDataProcess(test_data):
    testList = []
    test_data = test_data.replace(['NR'], [0.0])
    test_data = np.array(test_data).astype(float)
    for i in range(0, 4320, 18):
        testList.append(test_data[i:i+18][9])
    return np.array(testList)


#用测试数据集测试模型
def testModel(x_test, weight, bias):
    f = open("ouput.csv", 'w', encoding='utf-8', newline="")
    csv_write = csv.writer(f)
    csv_write.writerow(["id", "value"])
    for i in range(len(x_test)):
        output = weight.dot(x_test[i]) + bias
        csv_write.writerow(["id_" + str(i), str(output)])
    f.close()


#将weight, bias保存到pre.csv当中
def save_Pre(weight, bias):
    f = open("pre.csv", 'w', encoding='utf-8')
    csv_write = csv.writer(f)
    csv_write.writerow([",".join(str(i) for i in list(weight)), str(bias)])


#测试
def test():
    # 读取测试数据集
    pre = pd.read_csv("pre.csv")
    preList = list(pre.replace(",", " "))
    weight = []
    for i in list(preList[0].split(",")):
        weight.append(float(i))
    bias = float(preList[1])
    test_data = pd.read_csv("test.csv", header=None, usecols=np.arange(2, 11).tolist())
    x_test = testDataProcess(test_data)
    print(x_test.shape)
    print(x_test)
    testModel(x_test, np.array(weight), bias)


#训练
def train():
    df = pd.read_csv("train.csv", usecols=np.arange(3, 27).tolist()) #np.arange(3, 27)生成一个3-27的一组数，tolist()是把其转换成列表形式[3,4,...,27]
    x_data, y_data, data = dataProcess(df)
    print(x_data.shape)
    print(x_data)
    print(y_data.shape)
    print(y_data)
    x_train, y_train = x_data[0:3200], y_data[0:3200]
    x_val, y_val = x_data[3200:3600], y_data[3200:3600]
    weight, bais = trainModel(x_train, y_train, 2000)
    # save_Pre(weight, bais)
    print("训练得到的模型的weight为{}".format(weight))
    print("训练得到的模型的bias为{}".format(bais))
    loss = validateModel(x_val, y_val, weight, bais)
    print("模型在验证集上的loss为{:.2f}".format(loss))


if __name__ == '__main__':
    #test()
    train()
