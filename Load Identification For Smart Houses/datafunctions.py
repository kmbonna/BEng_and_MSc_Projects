#!/usr/bin/env python
# coding: utf-8

# In[2]:


from util.helper_functions import *
from appliance_type import *
import time
#import logging, sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
from os import listdir
from os.path import isfile, join
from collections import Counter
from numpy import argmax
from sklearn.preprocessing import LabelEncoder,LabelBinarizer
from timeit import default_timer as timer


appliance = ApplianceType()


# In[3]:


import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import math
import matplotlib
from sklearn.metrics import classification_report, confusion_matrix


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format= 'retina'")

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

colour_palette = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(colour_palette))

rcParams['figure.figsize']= 14,8


# In[8]:


def GetDictIDs(dictionary): 
    return list(dictionary["app_pws"].keys())


# In[1]:


def GetAppInstances(dictionary,appliance_id):
    return list(dictionary["app_pws"][appliance_id].keys())


# In[9]:


def GetSequence(dictionary,appliance_id, instance):    #x is dict
    return np.array(list(dictionary["app_pws"][appliance_id][instance]))  # returns array


# In[8]:


def RemoveNaN(seq):    # where seq is a numpy array
    return seq[~np.isnan(seq)]


# In[485]:


def MeanRigidThresholding(seq): #seq is a numpy array
    return seq[seq>np.mean(seq)]


# In[165]:


def RigidThresholding(seq,appliance_id):
    on_sequence = []
    thresholded = []
    turnon = False
    for dp in seq:
        if dp>= Thresholds[appliance_id]:                 #significant readings
            turnon = True
            on_sequence.append(dp) 
        else:
            if turnon == True:
                thresholded.append(on_sequence) #insignificant readings
            turnon = False    
            on_sequence = []
    return thresholded


# In[2]:


def InitialWindow(seq,appliance_id,threshold,width): #takes the whole two week period sequence
    windowed_sequences = []
    i = 0
    mean = 2 * np.mean(seq)
    while i < len(seq):
        try:
            if seq[i]>=threshold:
                reading = list(seq[i:i+width])
                if len(reading)<width:
                    reading.extend([0] * (width - len(reading)))
                windowed_sequences.append(reading)
                i+=width
            else:
                i+=1
        except KeyError:
            windowed_sequences = 0
            i = len(seq) # to stop the while loop
            continue
                
    return windowed_sequences


# In[1]:


def CutHalf(wseq,appliance_id, max_cut):              # takes one sequence and cut it in half if possible

    half = wseq[len(wseq)//2 : -1]
    if any(x<Thresholds[appliance_id] for x in half):
        decreased_window = wseq[ 0 : len(wseq)//2]
        decreased = True
    else:
        decreased_window = wseq
        decreased = False
    
    if len(decreased_window) < max_cut: 
        decreased = False
        
    return decreased_window, decreased


# In[2]:


def DecreaseWindows(sequences,appliance_id,max_cut):    # takes all the windowed sequences and cut them in half if possible
    dec = []
    for i in range(len(sequences)):
        decreased = True
        shorter = sequences[i]
        while decreased==True:
            shorter,decreased = CutHalf(shorter,appliance_id,max_cut)
        dec.append(shorter)
       
    return dec


# In[ ]:


def ClassicalFeatureExtraction(sequences):
    extracted_features = []
    
    for seq in sequences:
        t_power = np.sum(seq)
        a_power = np.mean(seq)
        max_power = np.max(seq)
        var = np.var(seq)
        extracted_features.append([seq[0], max_power, t_power, a_power, var])
        
    return extracted_features


# In[665]:


def flatten(t):
    return [item for sublist in t for item in sublist]


# In[1]:


def InvertEncoding(onehot_encoding, alphabet):
    inverts  = []
    int_to_id = dict((i, c) for i, c in enumerate(alphabet))

    # invert encoding
    
    for encoding in onehot_encoding:
        
        inverted = int_to_id[argmax(encoding)]
        
        inverts.append(inverted)
    
    return inverts


# In[1]:


def Split(Data):

    temp_train_imported_features = []
    temp_valid_imported_features = []
    temp_test_imported_features = []

    temp_train_imported_ids = []
    temp_valid_imported_ids = []
    temp_test_imported_ids = []

    for house in Data.keys():
        #making sure some appliances are in the validation and testing set
        if  (house=='101'):
            temp_valid_imported_features.append([datapoint[0] for datapoint in Data[house]])
            temp_valid_imported_ids.append([datapoint[1] for datapoint in Data[house]])
            continue

        elif (house=='103'):
            temp_test_imported_features.append([datapoint[0] for datapoint in Data[house]])
            temp_test_imported_ids.append([datapoint[1] for datapoint in Data[house]])
            continue


        split = np.random.choice(["training","validation", "testing"], 1, p=[0.7,0.15, 0.15])

        if split =="training":
            temp_train_imported_features.append([datapoint[0] for datapoint in Data[house]])
            temp_train_imported_ids.append([datapoint[1] for datapoint in Data[house]])

        if split =="validation":
            temp_valid_imported_features.append([datapoint[0] for datapoint in Data[house]])
            temp_valid_imported_ids.append([datapoint[1] for datapoint in Data[house]])

        if split =="testing":
            temp_test_imported_features.append([datapoint[0] for datapoint in Data[house]])
            temp_test_imported_ids.append([datapoint[1] for datapoint in Data[house]])


    # flattening out the list of lists
    train_imported_features = flatten(temp_train_imported_features)
    train_imported_ids = flatten(temp_train_imported_ids)
    valid_imported_features = flatten(temp_valid_imported_features)
    valid_imported_ids = flatten(temp_valid_imported_ids)
    test_imported_features = flatten(temp_test_imported_features)
    test_imported_ids = flatten(temp_test_imported_ids)
    
    return train_imported_features,train_imported_ids,valid_imported_features,valid_imported_ids,test_imported_features,test_imported_ids


# In[92]:


def EncodeLabels(labels):
    names= []
    
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    return encoded_labels, label_encoder.classes_


# In[82]:


def DecodeLabels(encoded_labels,encoded_label_classes):
    id_ = encoded_label_classes[encoded_labels]
    
    return appliance.get_appliance_name(id_)


# In[ ]:


#input is the feature and targets, output is sequences and labels
def Bootstrap(features,targets): 
    
    classes_count = Counter(targets)
    largest_class = np.max(list(classes_count.values()))

    d_unbalanced_ordered_sequences = {}

    bootstrapped_class_sequence = []
    bootstrapped_class_label = []

    for labels in classes_count.keys():
        #sorting the training data for each class label to be able to bootstrap
        indexes = [i for i in range(len(targets)) if targets[i] == labels]
        features_per_class = [features[i] for i in indexes]
        d_unbalanced_ordered_sequences[labels] = features_per_class 
    
        #bootstrapping
        indices = np.random.choice(np.arange(len(d_unbalanced_ordered_sequences[labels])), largest_class)
        temp = list(d_unbalanced_ordered_sequences[labels])
        bootstrapped_class_sequence.append([temp[i] for i in indices])
        bootstrapped_class_label.append([labels] * largest_class)
    
    boot_sequences= []
    boot_labels = []

    for i in range(len(bootstrapped_class_sequence)):
        for j in range(len(bootstrapped_class_sequence[i])):
            boot_sequences.append(bootstrapped_class_sequence[i][j])
            boot_labels.append(bootstrapped_class_label[i][j])
    
    return boot_sequences,boot_labels


# In[ ]:


def OneHotEncode(labels):
    one_hot_encoder = LabelBinarizer()
    temp_hotencoded_lst = one_hot_encoder.fit_transform(labels)
    hotencoded_array = np.array(temp_hotencoded_lst)
    return hotencoded_array, one_hot_encoder.classes_


# In[ ]:


# univariate for cnn
def Univariate(multi_sequences):
    temp_lst = [multi_sequences[i][0][0] for i in range(len(multi_sequences))]
    univariate_3d_array = np.expand_dims(np.array([np.array(xi) for xi in temp_lst]),axis=2)
    return univariate_3d_array


# In[87]:


def AddFeaturestofeatures(train_sequences,width,overlap=False): #adds the additional statistical measurement features
    
    f = []
    if overlap==False:
        for i in range(len(train_sequences)):
            t_power = []
            a_power = []
            max_power = []
            var = []
            if (i % int(len(train_sequences)/10)) ==0:
                print((i/len(train_sequences)) * 100, "%")
            j=0

            while j < len(train_sequences[0]):
            #for j in range(len(train_sequences[0][0])):
                t_power.append(float(np.sum(train_sequences[i][j:j+width])))
                a_power.append(float(np.mean(train_sequences[i][j:j+width])))
                max_power.append(float(np.max(train_sequences[i][j:j+width])))
                var.append(float(np.var(train_sequences[i][j:j+width])))
                
                j+=width

            f.append([train_sequences[i], t_power, a_power, max_power, var])
        
    elif overlap==True:
        for i in range(len(train_sequences)):

            t_power = []
            a_power = []
            max_power = []
            var = []
            
            if (i % int(len(train_sequences)/10)) ==0:
                print((i/len(train_sequences)) * 100, "%")

            for j in range(len(train_sequences[0])):
                t_power.append(np.sum(train_sequences[i][j:j+width]))
                a_power.append(np.mean(train_sequences[i][j:j+width]))
                max_power.append(np.max(train_sequences[i][j:j+width]))
                var.append(np.var(train_sequences[i][j:j+width]))

            f.append([train_sequences[i], t_power, a_power, max_power, var])
        
    return f


# In[ ]:


def AddFeaturestodataset(train_sequences,width,overlap):
    
    f = []
    dataset = []
    
    if overlap==False:
        for i in range(len(train_sequences)):
            t_power = []
            a_power = []
            max_power = []
            var = []
            if i % 10000==0:
                print((i/len(train_sequences)) * 100, "%")
            j=0

            while j < len(train_sequences[0][0]):
            #for j in range(len(train_sequences[0][0])):
                t_power.append(np.sum(train_sequences[i][0][j:j+width]))
                a_power.append(np.mean(train_sequences[i][0][j:j+width]))
                max_power.append(np.max(train_sequences[i][0][j:j+width]))
                var.append(np.var(train_sequences[i][0][j:j+width]))
                j+=width

            f.append([train_sequences[i][0], t_power, a_power, max_power, var])
        
    elif overlap==True:
        for i in range(len(train_sequences)):
            t_power = []
            a_power = []
            max_power = []
            var = []
            if i % 10000==0:
                print((i/len(train_sequences)) * 100, "%")
                
            for j in range(len(train_sequences[0][0])):
                t_power.append(np.sum(train_sequences[i][0][j:j+width]))
                a_power.append(np.mean(train_sequences[i][0][j:j+width]))
                max_power.append(np.max(train_sequences[i][0][j:j+width]))
                var.append(np.var(train_sequences[i][0][j:j+width]))

            f.append([train_sequences[i][0], t_power, a_power, max_power, var])
    
    for i in range(len(f)):
        dataset.append((f[i], train_sequences[i][1]))
        
    return dataset


# In[ ]:


# it is necessary for these functions to be performed in that order
def undersample(class_id):
    return midpoint if class_counter[class_id] > midpoint else class_counter[class_id]

def oversample(class_id):
    return midpoint


# In[ ]:


def show_confusion_matrix(confusion_matrix):
    hmap = sns.heatmap(confusion_matrix,annot=True,fmt="d",cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(),rotation=0,ha="right")
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(),rotation=30,ha="right")
    plt.title("Confusion matrix of the validation data")
    plt.ylabel("True Appliance")
    plt.xlabel("Predicted Appliance")

