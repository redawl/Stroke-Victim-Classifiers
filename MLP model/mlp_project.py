import pandas as pd
import numpy as np
import csv
import random
import copy
import matplotlib.pyplot as plt

# training and test set file name
train_file = "train_new.csv"
test_file = "test_new.csv"

# read csv file and create a dataframe
df_trainingset = pd.read_csv(train_file, sep=",")
df_testingset = pd.read_csv(test_file, sep=",")

train_label = df_trainingset['stroke'].tolist()
train_data = df_trainingset.drop(df_trainingset.iloc[:,0:1],axis=1)
train_data = train_data.values

test_label = df_testingset['stroke'].tolist()
test_data = df_testingset.drop(df_testingset.iloc[:,0:1],axis=1)
test_data = test_data.values

# multiperceptron class
class MULTIPERCEPTON:
    def __init__(self,train_label,train_data,test_label,test_data,hiddenneurons,outputneurons,eta, epochs):
        self.train_label = train_label
        self.train_data = train_data
        self.test_label = test_label
        self.test_data = test_data
        self.train_datalabel_count = self.train_data.shape[0]
        self.input_layer_neurons = self.train_data.shape[1]
        self.test_datalabel_count = self.test_data.shape[0]
        self.hidden_layer_neurons = hiddenneurons
        self.output_layer_neurons = outputneurons
        self.eta = eta
        self.max_val = 0.5
        self.min_val = -0.5
        self.hweight_matrix = None
        self.hbias_weight_matrix = None
        self.houtput_matrix = None
        self.oweight_matrix = None
        self.obias_weight_matrix = None
        self.ooutput_matrix = None
        self.target_matrix = None
        self.epochs = epochs
        self.training_correctness = 0
        self.training_incorrectness = 0
        self.hidden_weights()
        self.output_weights()
        self.onehot_encoding()

        self.test_houtput_matrix = None
        self.test_ooutput_matrix = None
        self.test_target_matrix = None
        self.testing_correctness = 0
        self.testing_incorrectness = 0
        self.test_onehot_encoding()
        
        self.test_target_list=[]
        self.test_predicted_list=[]

# activation function        
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
        
    def hidden_weights(self):
        H = self.input_layer_neurons * self.hidden_layer_neurons
        range_size = (self.max_val - self.min_val)  
        hweight = np.random.rand(H) * range_size + self.min_val

        # h weights matrix
        self.hweight_matrix = hweight.reshape(self.input_layer_neurons,self.hidden_layer_neurons)
        
        # bias value wt matrix
        hbias_weight = np.random.rand(self.hidden_layer_neurons) * range_size + self.min_val
        self.hbias_weight_matrix = hbias_weight.reshape(1,hbias_weight.shape[0])


    def output_weights(self):
        train_hidden_features = self.hidden_layer_neurons 
        OP = train_hidden_features * self.output_layer_neurons
        range_size = (self.max_val - self.min_val)  
        oweight = np.random.rand(OP) * range_size + self.min_val

        # o/p weights matrix
        self.oweight_matrix = oweight.reshape(train_hidden_features,self.output_layer_neurons)

        # o/p neuron matrix
        self.obias_weight_matrix = np.random.rand(self.output_layer_neurons) * range_size + self.min_val 
        self.obias_weight_matrix = np.array(self.obias_weight_matrix).reshape(1,self.obias_weight_matrix.shape[0])
        

    def forward_phase_houtput(self, index_i):
        self.input_feature = np.array(self.train_data[index_i]).reshape(1,self.train_data[index_i].shape[0])
        random.shuffle(self.input_feature)
        self.houtput_matrix = np.dot(self.input_feature,self.hweight_matrix) + self.hbias_weight_matrix
        self.houtput_matrix = self.sigmoid(self.houtput_matrix)


    def forward_phase_ooutput(self):
        self.ooutput_matrix = np.dot(self.houtput_matrix, self.oweight_matrix) + self.obias_weight_matrix
        self.ooutput_matrix = self.sigmoid(self.ooutput_matrix)
        
    
    def forward_phase_testhoutput(self, index_i):
        self.test_input_feature = np.array(self.test_data[index_i]).reshape(1,self.test_data[index_i].shape[0])
        random.shuffle(self.test_input_feature)
        self.test_houtput_matrix = np.dot(self.test_input_feature,self.hweight_matrix) + self.hbias_weight_matrix
        self.test_houtput_matrix = self.sigmoid(self.test_houtput_matrix)


    def forward_phase_testooutput(self):
        self.test_ooutput_matrix = np.dot(self.test_houtput_matrix, self.oweight_matrix) + self.obias_weight_matrix
        self.test_ooutput_matrix = self.sigmoid(self.test_ooutput_matrix)


#### Back propagation#######

# one hot encoding
    def onehot_encoding(self):
        labels = np.array([0]*self.train_datalabel_count)
        self.target_matrix = np.zeros((self.train_datalabel_count,self.output_layer_neurons), dtype=float)
        for i in range(self.train_datalabel_count):
            self.target_matrix[i, labels[i]] = self.train_label[i]
            train_label_value = self.train_label[i]
            for j in range(0,self.output_layer_neurons):
                if j == int(train_label_value):
                    self.target_matrix[i,j] = 0.9 
                else:
                    self.target_matrix[i,j] = 0.1 
    
    def test_onehot_encoding(self):
        test_labels = np.array([0]*self.test_datalabel_count)
        self.test_target_matrix = np.zeros((self.test_datalabel_count,self.output_layer_neurons), dtype=float)
        for i in range(self.test_datalabel_count):
            self.test_target_matrix[i, test_labels[i]] = self.test_label[i]
            test_label_value = self.test_label[i]
            for j in range(0,self.output_layer_neurons):
                if j == int(test_label_value):
                    self.test_target_matrix[i,j] = 0.9 
                else:
                    self.test_target_matrix[i,j] = 0.1 
    
    
    def calculate_accuracy(self, index_i):
        self.target_values = self.target_matrix[index_i].tolist()
        self.predicted_values = self.ooutput_matrix[0].tolist()
        
        target_index = self.target_values.index(max(self.target_values))
        predicted_index = self.predicted_values.index(max(self.predicted_values))

        if target_index == predicted_index:
            self.training_correctness += 1
        else:
            self.training_incorrectness += 1
        
    def get_epoch_accuracy(self):
        input_label_count = len(self.train_label)
        training_accuracy = (self.training_correctness/input_label_count)*100
        self.training_correctness = 0
        self.training_incorrectness = 0
        return(training_accuracy)
    
    def calculate_testaccuracy(self, index_i, epoch_cnt):
        self.test_target_values = self.test_target_matrix[index_i].tolist()
        self.test_predicted_values = self.test_ooutput_matrix[0].tolist()
        
        test_target_index = self.test_target_values.index(max(self.test_target_values))
        test_predicted_index = self.test_predicted_values.index(max(self.test_predicted_values))

        # list maintained to find confusion_matrix
        if epoch_cnt == self.epochs-1:
            self.test_target_list.append(test_target_index)
            self.test_predicted_list.append(test_predicted_index)
    
        if test_target_index == test_predicted_index:
            self.testing_correctness += 1
        else:
            self.testing_incorrectness += 1
            
        
    def get_testepoch_accuracy(self):
        test_input_label_count = len(self.test_label)
        test_accuracy = (self.testing_correctness/test_input_label_count)*100
        self.testing_correctness = 0
        self.testing_incorrectness = 0
        return(test_accuracy)
        
    
    def weight_updates(self, index_i):
        delta_o_t = (self.target_matrix[index_i] - self.ooutput_matrix)
        deltao = self.ooutput_matrix * (1-self.ooutput_matrix) * delta_o_t

        delta_h_o = np.dot(deltao,np.transpose(self.oweight_matrix))
        deltah = self.houtput_matrix * (1-self.houtput_matrix) * delta_h_o

        update_oweight = eta*(np.dot(np.transpose(self.houtput_matrix),deltao))
        update_hweight = eta*(np.dot(np.transpose(self.input_feature),deltah))
        
        # bias wt update
        update_hbweight = eta * deltah
        update_obweight = eta * deltao

        self.hweight_matrix = self.hweight_matrix + update_hweight
        self.oweight_matrix = self.oweight_matrix + update_oweight

        self.hbias_weight_matrix = self.hbias_weight_matrix + update_hbweight 
        self.obias_weight_matrix = self.obias_weight_matrix + update_obweight

    # confusion matrix    
    def cmatrix(self):
        a = self.test_target_list
        b = self.test_predicted_list
        cmatrix = np.zeros([self.output_layer_neurons, self.output_layer_neurons], dtype = int)
        range_val = len(self.test_label)

        for i in range(range_val):
            cmatrix[a[i],b[i]] +=1
            
        print()
        print(f"Confusion matrix for test set of {self.test_datalabel_count} data labels, (Hidden units = {self.hidden_layer_neurons})")
        print(cmatrix)
        

hiddenneurons = 4
outputneurons = 2 
eta = 0.1
epochs = 50
correct_train_accuracy = []
correct_test_accuracy = []

mlp = MULTIPERCEPTON(train_label,train_data,test_label,test_data,hiddenneurons,outputneurons,eta,epochs)

for epoch in range(epochs):
    for i in range(len(train_label)):
        mlp.forward_phase_houtput(i) 
        mlp.forward_phase_ooutput()
        mlp.calculate_accuracy(i)
        mlp.weight_updates(i)   
    trained_accuracy = mlp.get_epoch_accuracy()
    correct_train_accuracy.append(trained_accuracy)

    for j in range(len(test_label)):
        mlp.forward_phase_testhoutput(j)
        mlp.forward_phase_testooutput()
        mlp.calculate_testaccuracy(j,epoch)
    tested_accuracy = mlp.get_testepoch_accuracy()
    correct_test_accuracy.append(tested_accuracy)

    if epoch == epochs-1:
        mlp.cmatrix()


# plot
plt.plot(correct_train_accuracy)
plt.plot(correct_test_accuracy)
plt.title(f"learning rate {eta}; ephocs = {epochs}; hiddenunits = {hiddenneurons}")
plt.xlabel("Epoch")
plt.ylabel("Accuracy %")
plt.show()