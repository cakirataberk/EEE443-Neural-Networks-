#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 11:53:37 2022

@author: ata berk çakır 21703127
"""

import h5py
import random
import numpy as np
from matplotlib.pyplot import imshow, show, subplot, figure, axis, plot, xlabel,ylabel,title,savefig
 
def q1():
    
    ####################################### Part A ###################################
    
    #read data from .h5 file
    filename = "data1.h5"
    
    with h5py.File(filename, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]
    
        # Get the data
        #data = list(f[a_group_key])
        data = np.array(f[a_group_key])
        
    def normalize (data):
        #turn rgb to graysclae
        data_gray = np.array(data)[:,0,:,:]*0.2126 + np.array(data)[:,1,:,:]*0.7152 + np.array(data)[:,2,:,:]*0.0722
        mean = np.mean(data_gray,axis=(1,2)) #find mean
        
        for i in range (len(mean)):
            data_gray[i,:,:] -= mean[i] #remove the mean pixel intensity of each image from itself
            
        std = np.std(data_gray)  # find std
        data_gray = np.clip(data_gray, - 3 * std, 3 * std)  # clip -+3 std
        #normalize and map to 0.1 - 0.9
        normalized = data_gray*(0.90 - 0.10)/(2*np.max(data_gray))
        normalized += (0.10 - np.min(normalized))
        return normalized
    
    def random_index():
        random_index = []
        for i in range(200):
            index =random.randint(0,10239)
            random_index.append(index)
        return random_index
        
    
    def show_images(data,random_pics):
        figure(figsize=(20,10))
        j=0
        for i in random_pics:
            j+=1
            img = (data[i]).T
            subplot(10, 20, j)
            imshow(img)
            axis("off")
        savefig("fig1")
        
    def show_gray_images(data,random_pics):
        figure(figsize=(20,10))
        j=0
        for i in random_pics:
            j+=1
            img = (data[i]).T
            subplot(10, 20, j)
            imshow(img,cmap="gray")
            axis("off")
        savefig("fig2")
        
    ################################### PART B ####################################
        
    def w0 (Lpre,Lpost):
        w0 = np.sqrt(6/(Lpre + Lpost))
        return w0
    
    def initialize_weights(Lin, Lhid):
        
        Lout = Lin
        np.random.seed(42) 
        
        W1 = np.random.uniform(-w0(Lin,Lhid),w0(Lin,Lhid), size = (Lin,Lhid))
        W2 = np.random.uniform(-w0(Lhid,Lout),w0(Lhid,Lout), size = (Lhid,Lout))
        b1 = np.random.uniform(-w0(1,Lhid),w0(1,Lhid), size = (1,Lhid))
        b2 = np.random.uniform(-w0(1,Lout),w0(1,Lout), size = (1,Lout))
        
        we = [W1,W2,b1,b2]
        
        return we
    
    def initialize_params(Lin,Lhid,lmbd, beta, rho):    
        params = [Lin,Lhid,lmbd,beta,rho]
        
        return params
    
    def sigmoid(x):
        y = 1 / (1 + np.exp(-x))
        return y
    
    def sigmoid_backward(x):
        d_sig = x*(1-x)
        return d_sig
    
    def forward(data, we):
        
        w1,w2,b1,b2 = we
    
        z1 = np.dot(data,w1) + b1 #first layer linear forward
        A1 = sigmoid(z1) #first layer activation
        z2 =  np.dot(A1,w2) + b2 #output linear forward
        output = sigmoid(z2) #output layer activation
    
        return A1, output
    
    def aeCost(we, data, params):
        
        Lin,Lhid,lmbd,beta,rho = params
        w1,w2,b1,b2 = we
        N = len(data)
        
        A1, output = forward(data, we)
        mean = np.mean(A1, axis=0)
        
        #calculate cost 
        average_squared_error = (1/(2*N))*np.sum((data-output)**2)
        tykhonov = (lmbd/2)*(np.sum(w1**2) + np.sum(w2**2))
        kl_divergence = beta*np.sum((rho*np.log(mean/rho))+((1-rho)*np.log((1-mean)/(1-rho))))
        J = average_squared_error + tykhonov + kl_divergence
        d_hid = (np.dot(w2,(-(data-output)*sigmoid_backward(output)).T)+ (np.tile(beta*(-(rho/mean.T)+((1-rho)/(1-mean.T))), (10240,1)).T)) * sigmoid_backward(A1).T   
        
        d_w1 = (1/N)*(np.dot(data.T,d_hid.T) + lmbd*w1)
        d_w2 = (1/N)*(np.dot((-(data-output)*sigmoid_backward(output)).T,A1).T + lmbd*w2)
        d_b1 = np.mean(d_hid, axis=1)
        d_b2 = np.mean((-(data-output)*sigmoid_backward(output)), axis=0)
        
        Jgrad = [d_w1,d_w2,d_b1,d_b2]
        
        return J, Jgrad
    
    
    def backward(data, lr_rate, we, params):
        #get gradients
        J, Jgrad = aeCost(we, data, params)
        #update weights 
        we[0] -= lr_rate*Jgrad[0]
        we[1] -= lr_rate*Jgrad[1]
        we[2] -= lr_rate*Jgrad[2]
        we[3] -= lr_rate*Jgrad[3]
        return J, we
    
    def train(data_gray,epoch,Lin,Lhid,lmbd, beta, rho,lr_rate):
        losses = []
        epochs = []
        data_flat = np.reshape(data_gray, (data_gray.shape[0],data_gray.shape[1]**2))
        we = initialize_weights(Lin,Lhid)
        params = initialize_params(Lin,Lhid,lmbd, beta, rho)
    
    
        for i in range (epoch):
            J, we = backward(data_flat, lr_rate, we, params)
            epochs.append(i)
            losses.append(J)
            print("Epoch: {} --------------> Loss: {} ".format(i+1,J))
    
        return we,losses,epochs
    
    def plot_losses(losses,epochs):
        xlabel("epoch")
        ylabel("loss")
        plot(epochs,losses)

    
    def plot_weights(we,name):
        w1,w2,b1,b2 = we
        figure(figsize=(18, 16))
        plot_shape = int(np.sqrt(w1.shape[1]))
        for i in range(w1.shape[1]):
            subplot(plot_shape,plot_shape,i+1)
            imshow(np.reshape(w1[:,i],(16,16)), cmap='gray')
            axis('off')
        savefig(name)
    ############################# PART A ###########################
    index = random_index()
    show_images(data,index)
    data_gray = normalize (data)
    show_gray_images(data_gray,index)
    ############################## PART C #############################        
    we,losses,epochs = train(data_gray,100,256,64,5e-4, 0.01, 0.03,0.7)
    """plot_losses(losses,epochs)"""
    plot_weights(we,"fig3")
    ############################# PART D ###############################
    ############ Lhid = 16, alpha = 0 ############################    
    we,losses,epochs = train(data_gray,2000,256,16,0, 0.01, 0.03,0.7)
    plot_weights(we,"fig4")
    ############ Lhid = 49, alpha = 0 ############################    
    we,losses,epochs = train(data_gray,2000,256,49,0, 0.01, 0.03,0.7)
    plot_weights(we,"fig5")
    ############ Lhid = 81, alpha = 0 ############################    
    we,losses,epochs = train(data_gray,2000,256,81,0, 0.01, 0.03,0.7)
    plot_weights(we,"fig6")
    ############ Lhid = 16, alpha = 1e-3 ############################    
    we,losses,epochs = train(data_gray,2000,256,16,1e-3, 0.01, 0.03,0.7)
    plot_weights(we,"fig7")
    plot_losses(losses,epochs)
    ############ Lhid = 49, alpha = 1e-3 ############################    
    we,losses,epochs = train(data_gray,2000,256,49,1e-3, 0.01, 0.03,0.7)
    plot_weights(we,"fig8")
    ############ Lhid = 81, alpha = 1e-3 ############################    
    we,losses,epochs = train(data_gray,2000,256,81,1e-3, 0.01, 0.03,0.7)
    plot_weights(we,"fig9")
    plot_losses(losses,epochs)
    ############ Lhid = 16, alpha = 1e-6 ############################    
    we,losses,epochs = train(data_gray,1000,256,16,1e-6, 0.01, 0.03,0.7)
    plot_weights(we,"fig10")
    ############ Lhid = 49, alpha = 1e-6 ############################    
    we,losses,epochs = train(data_gray,1000,256,49,1e-6, 0.01, 0.03,0.7)
    plot_weights(we,"fig11")
    ############ Lhid = 81, alpha = 1e-6 ############################    
    we,losses,epochs = train(data_gray,1000,256,81,1e-6, 0.01, 0.03,0.7)
    plot_weights(we,"fig12")

 
    
def q2(): 
    filename = "data2.h5"
    with h5py.File(filename, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        y_test = f[list(f.keys())[0]][:]
        x_test = f[list(f.keys())[1]][:]
        y_train = f[list(f.keys())[2]][:]
        x_train = f[list(f.keys())[3]][:]
        y_val = f[list(f.keys())[4]][:]
        x_val = f[list(f.keys())[5]][:]
        words = f[list(f.keys())[6]][:]
        
    def word_vector(x, maxInd = 250):
        out = np.zeros(maxInd)
        out[x-1] = 1
        return out

    def input_vector(data):
        encoded_data = []
        for row in data:
            word_1 = word_vector(row[0])
            word_2 = word_vector(row[1])
            word_3 = word_vector(row[2])
            row = np.concatenate((word_1,word_2,word_3))
            row = row.reshape(1,len(row))
            encoded_data.append(row)
        return encoded_data

    def output_vector(data):
        encoded_data = []
        for row in data:
            word = word_vector(row)
            encoded_data.append(word)
        return encoded_data

    def initialize_weights(D,P,data,mean=0,std=0.01):
        N = 200
        W0 = np.random.normal(mean,std, 750*D).reshape(750, D)
        W1 = np.random.normal(mean,std, D*P).reshape(D,P)
        W2 = np.random.normal(mean,std, P*250).reshape(P,250)
        b1 = np.random.normal(mean,std, N*P).reshape(N,P)
        b2 = np.random.normal(mean,std, N*250).reshape(N,250)
        
        we = [W0,W1,W2,b1,b2]
        
        return we

    def sigmoid(x):
        y = 1 / (1 + np.exp(-x))
        return y

    def sigmoid_backward(x):
        d_sig = x*(1-x)
        return d_sig

    def softmax(x):
        expx = np.exp(x - np.max(x))
        y = expx/np.sum(expx, axis=0)
        return y

    def cross_entrophy(y,y_pred):
        return (np.sum(- y * np.log(y_pred))/ y.shape[0])

    def forward(data, we):
        
        w0,w1,w2,b1,b2 = we

        z0 = np.dot(data,w0) #first layer linear forward
        A0 = z0 #first layer without activation
        z1 = np.dot(A0,w1) + b1 #second layer linear forward
        A1 = sigmoid(z1) #second layer activation
        z2 =  np.dot(A1,w2) + b2 #output linear forward
        output = softmax(z2) #output layer activation

        return A0,A1,output

    def calculate_cost(data,y,we):
        N = data.shape[0]
        W0,W1,W2,b1,b2 = we
        A0,A1,output = forward(data, we)
        d_sig = sigmoid_backward(A1)
        #calculate cost
        cost = cross_entrophy(y,output)
        #calculate gradients
        d_w0 = np.dot(data.T,((np.dot(((np.dot((y-output),W2.T))*d_sig),W1.T))*A0))
        d_w1 = np.dot(A0.T,(np.dot((y-output),W2.T)*d_sig))
        d_w2 = np.dot(A1.T,(y-output))
        d_b1 = np.dot((y-output),W2.T)*d_sig
        d_b2 = y-output
        
        grads = [d_w0,d_w1,d_w2,d_b1,d_b2]
        
        return cost, grads

    def backward(data,y,lr_rate,momentum, we, old_grads):
        #get gradients
        cost, grads = calculate_cost(data,y,we)
        #update weights 
        we[0] -= lr_rate*grads[0]+old_grads[0]*momentum
        we[1] -= lr_rate*grads[1]+old_grads[1]*momentum
        we[2] -= lr_rate*grads[2]+old_grads[2]*momentum
        we[3] -= lr_rate*grads[3]+old_grads[3]*momentum
        we[4] -= lr_rate*grads[4]+old_grads[4]*momentum

        return cost, grads, we
        
    def train(x_train,y_train,D,P,epoch,num_batch,lr_rate,momentum):
        costs = []
        epochs = []
        data = input_vector(x_train)
        data = np.squeeze(data,axis=1)
        label = output_vector(y_train)
        label = np.array(label)
        we = initialize_weights(D,P,data,mean=0,std=0.01)
        momentum = 0.0000085
        cost, old_grads = calculate_cost(data[:200],label[0:200],we)
        lr_rate = 0.0000015
        for i in range (epoch):
            epochs.append(i)
            for j in range (num_batch):
                data_batch = data[200*j:200*j+200]
                label_batch = label[200*j:200*j+200]
                cost, grads,we = backward(data_batch,label_batch, lr_rate,momentum,we,old_grads)
            costs.append(cost)
        j = 0
        for i in reversed (costs):
            print("Epoch: {} --------------> Loss: {} ".format(j+1,i))
            j+=1
        return we,costs[::-1],epochs

    def random_index(sample_size):
        random_index = []
        for i in range(sample_size):
            index = random.randint(0,46500)
            random_index.append(index)
        return random_index

    def pick_sample(data,label,sample_size):
        sample = []
        labels = []
        sample_index = random_index(sample_size)
        for i in sample_index:
            sample.append(data[i])
            labels.append(label[i])
        return sample,labels

    def predict(words,output):
        pred_rows = []
        for i in range(len(output)):
            word_index = output[i].argsort()[-10:][::-1]
            pred_words = []
            for word in word_index: 
                pred_words.append(str(words[word].decode("utf-8")))
            pred_rows.append((pred_words))
        return pred_rows

    def print_preds(random_sample,test_label,pred_rows,words):
        for i in range(len(random_sample)):
            tri = "sample trigram: "
            for j in range(len(random_sample[i])):
                tri+=str(words[random_sample[i][j]].decode("utf-8"))+" "
            tri += " ----> label: " + str(words[test_label[i]].decode("utf-8"))
            print(tri)
            print("Top 10 predictions: ",pred_rows[i])
            
    def plot_losses(losses,epochs,titles):
        xlabel("epoch")
        ylabel("loss")
        title(titles)
        plot(epochs,losses)
    ########################## Test with different D and P values ######################
    
    """epoch changed to 5 in order to save tim TA while running main idea is to show code is running
    (default is 30 if you want to try you can change by yourself 5th parameter)"""
    
    we,costs,epochs = train(x_train,y_train,32,256,5,1000,0.15,0.85)
    """we,costs,epochs = train(x_val,y_val,32,256,30,232,0.15,0.85)"""
    plot_losses(costs,epochs,"D=32 - P=256")
    
    """this part is commented in order to save time for TA to while running
    
    we,costs,epochs = train(x_train,y_train,16,128,30,1000,0.15,0.85)
    we,costs,epochs = train(x_val,y_val,16,32,30,232,0.15,0.85)
    plot_losses(costs,epochs,"D=16 - P=128")
    
    we,costs,epochs = train(x_train,y_train,8,64,30,1000,0.15,0.85)
    we,costs,epochs = train(x_val,y_val,16,32,30,232,0.15,0.85)
    plot_losses(costs,epochs,"D=8 - P=64")"""
    
    ######################## Make Predictions with Test Data ##############################
    random_sample,test_label = pick_sample(x_test,y_test,200)
    random_sample_vector = input_vector(random_sample)
    random_sample_vector = np.squeeze(random_sample_vector,axis=1)
    random_sample = random_sample[:5]
    test_label  = test_label[:5]
    
    _,_,output = forward(random_sample_vector, we)
    output = output[:5]
    
    pred_rows = predict(words,output)
    print_preds(random_sample,test_label,pred_rows,words)


import sys

question = sys.argv[1]

def ataberk_cakir_21703127_hw1(question):
    if question == '1' :
        print("Question 1")
        q1()
    elif question == '2' :
        print("Question 2")
        q2()

ataberk_cakir_21703127_hw1(question)

    


    
    