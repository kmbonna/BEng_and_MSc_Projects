#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from datafunctions import *
import pickle
import pandas as pd
from keras.preprocessing.sequence import pad_sequences


# ## Load the Data

# ### If already Split Training-Validation Data

# #### Load Training data

# In[ ]:


# Change the directory
path = "C:/Users/Karim/uk-ucl-2021/5Min"
os.chdir(path)


# In[ ]:


Data_np = np.load("Data_Per_House.npy", allow_pickle=True)
Data = Data_np.item()


# In[ ]:


train_imported_features,train_imported_ids,valid_imported_features,valid_imported_ids,test_imported_features,test_imported_ids = Split(Data)


# In[ ]:


# check if all is included
print(np.unique(valid_imported_ids))
print(np.unique(train_imported_ids))
print(np.unique(test_imported_ids))


# In[ ]:


#train_imported_features = np.load("Training_Features.npy", allow_pickle=True).tolist()


# In[ ]:


#train_imported_ids = np.load("Training_IDs.npy", allow_pickle=True).tolist()


# In[ ]:


all_count = {}
for i, j in count1.items():
    for x, y in count2.items():
        for a,b in count3.items():
            if (i == x == a):
                
                all_count[i]=(j+y+b)


# In[ ]:


#count1 = Counter(train_imported_ids)
#count2 = Counter(valid_imported_ids)
#count3 = Counter(test_imported_ids)


# In[ ]:


#np.unique(train_imported_ids)


# In[ ]:


#total_instances = 0
#house_count =[]
#hous = []
#for house in all_count.keys():
#    house_count.append(all_count[house])
#    hous.append(appliance.get_appliance_name(house))
#    total_instances += all_count[house]
    

#plt.bar(hous,house_count) 
#plt.title("sequences per appliance in the dataset")
#plt.xlabel("appliances")
#plt.ylabel("sequences")
#plt.tight_layout()

#plt.xticks(hous, rotation='50')

#plt.show()


# In[ ]:


#total_instances = 0
#house_count =[]
#hous = []
#for house in Data.keys():
#    house_count.append(len(list(Data[house])))
#    hous.append(house)
#    total_instances += len(list(Data[house]))
    

#plt.bar(hous,house_count) 
#plt.title("relevant houses sequences in the dataset")
#plt.xlabel("houses")
#plt.ylabel("instances")
#plt.tight_layout()

#plt.xticks(hous, rotation='90')

#plt.show()


# #### Load Validation data

# In[ ]:


#valid_imported_features = np.load("Validation_Features.npy", allow_pickle=True).tolist()


# In[ ]:


#valid_imported_ids = np.load("Validation_IDs.npy", allow_pickle=True).tolist()


# #### Load Testing data

# In[ ]:


#test_imported_features = np.load("Testing_Featuresv2.npy", allow_pickle=True).tolist()


# In[ ]:


#test_imported_ids = np.load("Testing_IDsv2.npy", allow_pickle=True).tolist()


# # Cleaning

# #### Perform OverSampling on training data

# In[ ]:


from imblearn.over_sampling import SMOTE,RandomOverSampler,ADASYN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler


# In[ ]:


class_counter = Counter(train_imported_ids)
class_counter


# In[ ]:


occurances = list(class_counter.values())
largest_occurance = np.max(occurances)
smallest_occruance = np.min(occurances)
print(largest_occurance,smallest_occruance)


# In[ ]:


midpoint = int((largest_occurance/ 3))
midpoint


# In[ ]:


# it is necessary for these functions to be performed in that order
def undersample(class_id):
    return midpoint if class_counter[class_id] > midpoint else class_counter[class_id]

def oversample(class_id):
    return midpoint


# In[ ]:


OverSampling= {}
UnderSampling= {}

for c in class_counter.keys():
    UnderSampling[c]= undersample(c)
    OverSampling[c] = oversample(c)


# In[ ]:


UnderSampler = RandomUnderSampler(sampling_strategy = UnderSampling)
temp_f, temp_ids = UnderSampler.fit_resample(train_imported_features,train_imported_ids)


# In[ ]:


check = Counter(temp_ids)
check


# In[ ]:


smote = SMOTE(sampling_strategy=OverSampling, k_neighbors=50)

train_oversampled_features, train_oversampled_ids = smote.fit_resample(temp_f, temp_ids)


# In[ ]:


check = Counter(train_oversampled_ids)
check


# #### Add Window-Extracted Features

# In[ ]:


window_width = 30
window_overlap = False


# In[ ]:


train_multi_features = AddFeaturestofeatures(train_oversampled_features,width=window_width,overlap=window_overlap)


# In[ ]:


valid_multi_features = AddFeaturestofeatures(valid_imported_features,width=window_width,overlap=window_overlap)


# In[ ]:


test_multi_features = AddFeaturestofeatures(test_imported_features,width=window_width,overlap=window_overlap)


# #### If non-overlapping or jump>1 , pad the features

# In[ ]:


x_train=[]
for x in train_multi_features:
    pad = pad_sequences(x, dtype='float32', padding='pre')
    x_train.append(pad)


# In[ ]:


x_valid=[]
for x in valid_multi_features:
    pad = pad_sequences(x, dtype='float32', padding='pre')
    x_valid.append(pad)


# In[ ]:


x_test=[]
for x in test_multi_features:
    pad = pad_sequences(x, dtype='float32', padding='pre')
    x_test.append(pad)


# #### If overlapping:

# In[ ]:


#x_train = train_multi_features
#x_valid= valid_multi_features
#x_test= test_multi_features


# #### If other features are not added

# In[ ]:


#x_train = train_oversampled_features
#x_valid= valid_imported_features
#x_test= test_imported_features


# #### Encoding

# Training

# In[ ]:


y_train, train_label_decoder = EncodeLabels(train_oversampled_ids)


# Validating

# In[ ]:


y_valid, valid_label_decoder = EncodeLabels(valid_imported_ids)


# Testing

# In[ ]:


y_test, test_label_decoder =  EncodeLabels(test_imported_ids)


# In[ ]:


np.unique(train_oversampled_ids)


# #### Combining x and y

# In[ ]:


train_sequences = [(x_train[i], y_train[i]) for i in range(len(x_train))]


# In[ ]:


valid_sequences = [(x_valid[j], y_valid[j]) for j in range(len(x_valid))]


# In[ ]:


test_sequences = [(x_test[j], y_test[j]) for j in range(len(x_test))]


# In[ ]:


len(train_sequences)


# And We're done

# ### If you want to save

# In[ ]:


# Change the directory
path = "C:/Users/Karim/uk-ucl-2021/5Min"
os.chdir(path)


# In[ ]:


len(train_sequences)


# In[ ]:


np.save("Training-Datafinal",train_sequences)


# In[ ]:


np.save("Validation-Datafinal",valid_sequences)
np.save("Testing-Datafinal",test_sequences)


# ## If there is an already made training and verification dataset load here:

# In[ ]:


# Change the directory
path = "C:/Users/Karim/uk-ucl-2021/5Min"
os.chdir(path)


# In[ ]:


train_sequences = np.load("Training-Datafinal.npy", allow_pickle=True).tolist()
valid_sequences = np.load("Validation-Datafinal.npy", allow_pickle=True).tolist()
test_sequences = np.load("Testing-Datafinal.npy", allow_pickle=True).tolist()


# In[ ]:


x_train = [train_sequences[i][0] for i in range(len(train_sequences))]
y_train = [train_sequences[i][1] for i in range(len(train_sequences))]


# In[ ]:


x_valid = [valid_sequences[i][0] for i in range(len(valid_sequences))]
y_valid = [valid_sequences[i][1] for i in range(len(valid_sequences))]


# In[ ]:


x_test = [test_sequences[i][0] for i in range(len(test_sequences))]
y_test = [test_sequences[i][1] for i in range(len(test_sequences))]


# In[ ]:


# if univariate wanted

#x_train = Univariate(train_sequences)
#y_train = [train_sequences[i][1] for i in range(len(train_sequences))]
#x_valid = Univariate(valid_sequences)
#y_valid = [valid_sequences[i][1] for i in range(len(valid_sequences))]
#x_test =  Univariate(test_sequences)
#y_test =  [test_sequences[i][1] for i in range(len(test_sequences))]


# 
# # LSTM

# In[ ]:


import pandas as pd
import numpy as np

from tqdm.auto import tqdm
from tqdm.notebook import tqdm
import pytorch_lightning as pl


from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import MinMaxScaler


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


from matplotlib.ticker import MaxNLocator
from multiprocessing import cpu_count
from pytorch_lightning.metrics.functional import accuracy
from sklearn.metrics import classification_report, confusion_matrix
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningModule, Trainer


# In[ ]:


print(torch.cuda.get_device_name(0))
#torch.cuda.memory_summary(device=None, abbreviated=False)


# In[ ]:


import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import math
import matplotlib

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format= 'retina'")

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

colour_palette = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(colour_palette))

rcParams['figure.figsize']= 14,8

#pl.seed_everything(42)


# ## PyTorch Dataset

# In[ ]:


class ApplianceTypeDataset(Dataset):
    
    def __init__(self,sequences):
        self.sequences = sequences
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self,idx):
        sequence, label = self.sequences[idx]
        
        sequence=np.array([np.array(xi) for xi in sequence])

        return dict(
        sequence=torch.tensor(np.array(sequence)).float(),
        label= torch.tensor(label).long()
        )


# In[ ]:


class ApplianceTypeDataModule(pl.LightningDataModule):
    
    def __init__(self, train_sequences,valid_sequences, batch_size):
        super().__init__()
        self.train_sequences = train_sequences
        self.valid_sequences = valid_sequences
        self.test_sequences = test_sequences
        self.batch_size = batch_size
    
    
    def train_dataloader(self):
        train_dataset = ApplianceTypeDataset(self.train_sequences)
        return DataLoader(
        train_dataset,
        batch_size= self.batch_size,
        shuffle=True,
        num_workers= 0,
        #persistent_workers=True,
        pin_memory=True
        )
    
    def val_dataloader(self):
        
        valid_dataset = ApplianceTypeDataset(self.valid_sequences)
        return DataLoader(
        valid_dataset,
        batch_size= self.batch_size,
        shuffle=True,
        num_workers= 0,
        #persistent_workers=True,
        pin_memory=True
        )
    
    def test_dataloader(self):
        
        test_dataset = ApplianceTypeDataset(self.test_sequences)
        return DataLoader(
        test_dataset,
        batch_size= self.batch_size,
        shuffle=False,
        num_workers= 0,
        #persistent_workers=True,
        pin_memory=True
        )   


# #### Hyperparameters

# In[ ]:


n_epoch = 20
batch_size= 128
learning_rate = 0.0001


# ## Model

# In[ ]:


class LSTM(nn.Module):
    def __init__(self,n_features,n_classes,n_hidden, n_layers,dropout):
        super().__init__()
        
        self.lstm = nn.LSTM(
        input_size= n_features,
        hidden_size=n_hidden,
        num_layers=n_layers,
        batch_first= True,
        dropout=dropout
        )
        
        self.classifier = nn.Linear(n_hidden,n_classes)
    
    def forward(self, x):
        self.lstm.flatten_parameters()
        
        _,(hidden,_) = self.lstm(x)
        
        out1 = hidden[-1]
        out2 = self.classifier(out1)

        return out2  


# In[ ]:


class AppliancePredictor(pl.LightningModule):
    def __init__(self, n_features: int, n_classes: int,n_hidden: int,n_layers: int, dropout: float):
        super().__init__()
        self.model = LSTM(n_features, n_classes,n_hidden,n_layers,dropout)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self,x, labels=None):
        #x = torch.unsqueeze(x,2)          # for univariate
        x = torch.swapaxes(x, 1,2)  # for multivariate
        output = self.model(x)
        
        loss = 0
        if labels is not None:
            loss = self.criterion(output,labels)
        return loss, output
    
    def training_step(self,batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]
        loss, outputs = self(sequences,labels)
        
        predictions = torch.argmax(outputs, dim=1)
        step_accuracy  = accuracy(predictions, labels)
        
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_accuracy", step_accuracy, prog_bar=True, logger=True)
        
        return {"loss": loss, "accuracy": step_accuracy}
    
    def validation_step(self,batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]
        loss, outputs = self(sequences,labels)
        
        predictions = torch.argmax(outputs, dim=1) 
        step_accuracy  = accuracy(predictions, labels)
        
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_accuracy", step_accuracy, prog_bar=True, logger=True)
        
        return {"loss": loss, "accuracy": step_accuracy}
    
    def test_step(self,batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]
        loss, outputs = self(sequences,labels)
        
        
        predictions = torch.argmax(outputs, dim=1) 
        step_accuracy  = accuracy(predictions, labels)
        
        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_accuracy", step_accuracy, prog_bar=True, logger=True)
        
        return {"loss": loss, "accuracy": step_accuracy}
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=learning_rate)  
    
    #def predict_step(self, batch, batch_idx: int , dataloader_idx: int = None):
    #    return self(batch)


# In[ ]:


# Change the directory
path = "C:/Users/Karim/uk-ucl-2021/5Min"
os.chdir(path)

checkpoint_callback = ModelCheckpoint(dirpath= "lstm+features-checkpoints",filename= "best-checkpoint",
                                      save_top_k=1,verbose=True,monitor="val_loss",mode="min"
                                     )

early_stopping_callback = EarlyStopping(monitor="val_loss", patience=4)

logger= TensorBoardLogger("logs", name="bootstrap_lstm")


# In[ ]:


print(torch.cuda.get_device_name(0))
#torch.cuda.memory_summary(device=None, abbreviated=False)


# In[ ]:


if len((x_train[0])) > 5:
    number_of_features = 1
else:
    number_of_features = len((x_train[0]))


# In[ ]:


hidden_nodes = 256
layers = 2
dp = 0.3


# In[ ]:


import multiprocessing as mp

if __name__ == '__main__':
    
    mp.set_start_method("spawn", force=True)
                
    trainer= pl.Trainer(num_sanity_val_steps=0,logger=logger,
                        callbacks = [early_stopping_callback, checkpoint_callback],
                        max_epochs=n_epoch, auto_select_gpus=True,
                        gpus=1,progress_bar_refresh_rate=10)
                
    data_module = ApplianceTypeDataModule(train_sequences,valid_sequences, batch_size)
    
    model = AppliancePredictor(n_features=number_of_features,n_classes=len(np.unique(y_train)),
                          n_hidden=hidden_nodes, n_layers=layers,dropout=dp)

    
    trainer.fit(model,data_module)
                
    #print("previous model was fit using",hidden_nodes,"nodes,",layers,"layers and dropout of",dp)


# ### Visualisation

# In[ ]:


# Change the directory
path = "C:/Users/Karim/uk-ucl-2021/5Min/lstm+features-checkpoints"
os.chdir(path)


# In[ ]:


trained_model = AppliancePredictor.load_from_checkpoint(
    checkpoint_path="best-checkpoint.ckpt",
    n_features=number_of_features,
    n_classes=len(np.unique(y_train)),
    n_hidden=hidden_nodes,
    n_layers=layers,
    dropout=dp
    
)
trained_model.freeze()


# In[ ]:


valid_dataset = ApplianceTypeDataset(valid_sequences)

predictions =[]
labels = []
wrong_ones = []
corr = []
p = []
for item in tqdm(valid_dataset):
    sequence = item["sequence"]
    label = item["label"]
    #app_id = torch.argmax(label,dim=1)
    _, output = trained_model(sequence.unsqueeze(dim=0))
    prediction= torch.argmax(output,dim=1)
    if prediction!=label:
        #print(output)
        p.append(prediction)
        corr.append(label)
        wrong_ones.append(sequence)
    predictions.append(prediction.item())
    labels.append(label.item())


# In[ ]:


#decoder = [2,5,6,20,24,101,112,116,131]


# In[ ]:


names = []
for t in valid_label_decoder:
    names.append(appliance.get_appliance_name(t))
names


# In[ ]:


print(classification_report(labels,predictions,target_names=names))


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels,predictions)
df_cm = pd.DataFrame(cm, index=names, columns=names)


# In[ ]:


show_confusion_matrix(df_cm)


# ## Testing

# In[ ]:


# Change the directory
path = "C:/Users/Karim/uk-ucl-2021/5Min/lstm+features-checkpoints"
os.chdir(path)


trained_model = AppliancePredictor.load_from_checkpoint(
    checkpoint_path="best-checkpoint.ckpt",
    n_features=number_of_features,
    n_classes=len(np.unique(y_train)),
    n_hidden=hidden_nodes,
    n_layers=layers,
    dropout=dp
    
)
trained_model.freeze()


# In[ ]:


test_dataset = ApplianceTypeDataset(test_sequences)

predictions =[]
labels = []
wrong_ones = []
corr = []
p = []
for item in tqdm(test_dataset):
    sequence = item["sequence"]
    label = item["label"]
    #app_id = torch.argmax(label,dim=1)
    _, output = trained_model(sequence.unsqueeze(dim=0))
    prediction= torch.argmax(output,dim=1)
    if prediction!=label:
        p.append(prediction)
        corr.append(label)
        wrong_ones.append(sequence)
    predictions.append(prediction.item())
    labels.append(label.item())


# In[ ]:


print(classification_report(labels,predictions,target_names=names))


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels,predictions)
df_cm = pd.DataFrame(cm, index=names, columns=names)


# In[ ]:


def show_confusion_matrix(confusion_matrix):
    hmap = sns.heatmap(confusion_matrix,annot=True,fmt="d",cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(),rotation=0,ha="right")
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(),rotation=30,ha="right")
    plt.title("Confusion matrix of the validation data")
    plt.ylabel("True Appliance")
    plt.xlabel("Predicted Appliance")


# In[ ]:


show_confusion_matrix(df_cm)


# ### Investigate wrong cases

# In[ ]:


np.where(test_label_encoder==24)


# In[ ]:


idx = np.where(np.array(corr)==4)[0]


# In[ ]:


idx[0:100]


# In[ ]:


#investigate = 5
#print("Correct ",appliance.get_appliance_name(test_label_encoder[corr[investigate]]))
#print("Predicted ",appliance.get_appliance_name(test_label_encoder[p[investigate]]))

#plt.plot(np.arange(300),wrong_ones[investigate][0])
#plt.title("check")
#plt.xlabel("time stamp")
#plt.ylabel("power reading")

#plt.tight_layout()


#plt.show()


# # 1DCNN

# In[ ]:


import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Conv1D,MaxPooling1D, Flatten, SeparableConv1D, AveragePooling1D
from keras.layers import LSTM,Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
from keras_tqdm import TQDMNotebookCallback
# fix random seed for reproducibility
#np.random.seed(7)
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


cnn_x_train = Univariate(train_sequences)
cnn_x_valid = Univariate(valid_sequences)
cnn_x_test = Univariate(test_sequences)


# In[ ]:


len(cnn_x_test)


# In[ ]:


def OneHotEncode(labels):
    one_hot_encoder = LabelBinarizer()
    temp_hotencoded_lst = one_hot_encoder.fit_transform(labels)
    hotencoded_array = np.array(temp_hotencoded_lst)
    return hotencoded_array, one_hot_encoder.classes_


# In[ ]:


cnn_y_train, trdecoder = OneHotEncode(train_oversampled_ids)
cnn_y_valid, vdecoder = OneHotEncode(valid_imported_ids)
cnn_y_test, tedecoder = OneHotEncode(test_imported_ids)


# In[ ]:


print(Counter(y_valid))


# In[ ]:


cnn_y_valid.shape


# In[ ]:


lst = []
for l in cnn_y_valid:
    lst.append(np.argmax(l))
    
validation_class_counter = Counter(lst)
validation_class_counter


# In[ ]:


class_weights= {}
for c in validation_class_counter.keys():
    class_weights[c]=  np.max(list(validation_class_counter.values()))/validation_class_counter[c]


# In[ ]:


class_weights


# In[ ]:


weights = []
for s in y_valid:
    weights.append(class_weights[s])
w = np.array(weights)   


# In[ ]:


# Change the directory
path = "C:/Users/Karim/uk-ucl-2021/5Min"
os.chdir(path)


# In[ ]:


filepath="weight_best_cnn_new_data.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss',save_top_k=1,verbose=True,
                             save_best_only=True, mode='min')
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

loss_function_used = CategoricalCrossentropy(from_logits=True)
optimizer_used = Adam(learning_rate=0.0005)
callbacks_list = [checkpoint,early_stop]

# create the model
model = Sequential()
model.add(Conv1D(filters=16, kernel_size=5, padding='valid', activation='relu'))
model.add(Dropout(0.1))
model.add(AveragePooling1D(pool_size=3))
model.add(Conv1D(filters=16, kernel_size=3, padding='valid', activation='relu'))
model.add(Dropout(0.1))
model.add(AveragePooling1D(pool_size=3))
model.add(Flatten())
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(len(cnn_y_train[0]), activation='linear'))
model.compile(loss=loss_function_used, optimizer=optimizer_used, metrics=['accuracy'])
model.fit(cnn_x_train, cnn_y_train, epochs=20, batch_size=128,verbose = 1,
          callbacks = callbacks_list,validation_data=(cnn_x_valid,cnn_y_valid))


# In[ ]:


model.load_weights("weight_best_cnn_new_data.hdf5")
scores = model.evaluate(cnn_x_valid, cnn_y_valid, verbose=True)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:


PREDS = []
LABELS =[]
cnn_output = model.predict(cnn_x_valid, batch_size=512)
preds = np.argmax(cnn_output, axis=1)
labels = np.argmax(cnn_y_valid, axis=1)


# In[ ]:


names = []
for t in vdecoder:
    names.append(appliance.get_appliance_name(t))
names


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(labels,preds,target_names=names))


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels,preds)
df_cm = pd.DataFrame(cm, index=names, columns=names)


# In[ ]:


show_confusion_matrix(df_cm)


# ## Testing

# In[ ]:


PREDS = []
LABELS =[]
cnn_output = model.predict(cnn_x_test, batch_size=512)
preds = np.argmax(cnn_output, axis=1)
labels = np.argmax(cnn_y_test, axis=1)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(labels,preds,target_names=names))


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels,preds)
df_cm = pd.DataFrame(cm, index=names, columns=names)
show_confusion_matrix(df_cm)


# ## investigating

# In[ ]:


print(train_label_decoder)
print(valid_label_decoder)


# ### tumble

# In[ ]:


aircon = np.where(np.array(y_train)==7)[0]
aircon2 = np.where(np.array(y_test)==7)[0]
len(aircon)


# In[ ]:


plt.subplot(1, 2, 1)
for n in aircon[0:175]:
    plt.plot(np.arange(300), x_train[n])
    plt.title("training data")

plt.subplot(1, 2, 2)
for n in aircon2[0:175]:
    plt.plot(np.arange(300), x_test[n])
    plt.title("valid data")


# ### Clothes washer

# In[ ]:


washer = np.where(y_train==1)[0]
washer2 = np.where(y_valid==1)[0]
len(washer)


# In[ ]:


plt.subplot(1, 2, 1)
for n in washer[0:175]:
    plt.plot(np.arange(300), x_train[n])
    plt.title("training data")

plt.subplot(1, 2, 2)
for n in washer2[50:60]:
    plt.plot(np.arange(300), x_valid[n])
    plt.title("valid data")


# In[ ]:


coffee = np.where(y_train==2)[0]
coffee2 = np.where(y_valid==2)[0]


# In[ ]:


for n in coffee[4000:5000]:
    plt.plot(np.arange(300), x_train[n])
    plt.title("training data")


# In[ ]:


for n in coffee2[0:50]:
    plt.plot(np.arange(300), x_valid[n])
    plt.title("valid data")


# In[ ]:


plt.subplot(1, 2, 1)
for n in coffee[0:15]:
    plt.plot(np.arange(300), x_train[n])
    plt.title("training data")

plt.subplot(1, 2, 2)
for n in coffee2[0:10]:
    plt.plot(np.arange(300), x_valid[n])
    plt.title("valid data")


# In[ ]:


dwasher = np.where(y_train==7)[0]
dwasher2 = np.where(y_valid==7)[0]
len(dwasher2)


# In[ ]:


plt.subplot(1, 2, 1)
for n in dwasher[0:100]:
    plt.plot(np.arange(300), x_train[n])
    plt.title("training data")

plt.subplot(1, 2, 2)
for n in dwasher2[0:100]:
    plt.plot(np.arange(300), x_valid[n])
    plt.title("valid data")


# In[ ]:


# for n in idx[7653:8000]:
    plt.plot(np.arange(300), x_train[n])


# #### CNN-LSTM

# In[ ]:


#filepath="weight_best_cnn_new_data.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='val_loss',save_top_k=1,verbose=True,
#                             save_best_only=True, mode='min')
#early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

#loss_function_used = CategoricalCrossentropy(from_logits=True)
#optimizer_used = Adam(learning_rate=0.0005)
#callbacks_list = [checkpoint,early_stop]



#max_length = 300
#model = Sequential()
#model.add(Conv1D(filters=16, kernel_size=7, padding='valid', activation='relu'))
#model.add(MaxPooling1D(pool_size=2))
#model.add(LSTM(100))
#model.add(Dense(len(cnn_y_train[0]), activation='linear'))
#model.compile(loss=loss_function_used, optimizer=optimizer_used, metrics=['accuracy'])
#model.fit(cnn_x_train, cnn_y_train, epochs=5, batch_size=128,verbose = 1,
#          callbacks = callbacks_list,validation_data=(cnn_x_valid,cnn_y_valid))
#print(model.summary())


# In[ ]:


#model.load_weights("weights_best_cnn_lstm.hdf5")
#scores = model.evaluate(cnn_x_test, cnn_y_test, verbose=True)
#print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:


#PREDS = []
#LABELS =[]
#cnn_output = model.predict(cnn_x_test, batch_size=128)
#preds = np.argmax(cnn_output, axis=1)
#labels = np.argmax(cnn_y_test, axis=1)


# In[ ]:


#print(classification_report(labels,preds,target_names=names))


# In[ ]:


#cm = confusion_matrix(labels,preds)
#df_cm = pd.DataFrame(cm, index=names, columns=names)

#show_confusion_matrix(df_cm)

