import numpy  as np
import torch
from tqdm import tqdm
import pickle


def load_dataset():
    with open('/home/hadoop/Desktop/features.pkl','rb') as f: # change the path to load dataset
        features = pickle.load(f)
    with open('/home/hadoop/Desktop/labels.pkl','rb') as f: # change the path to load dataset
        labels = pickle.load(f)


    split_train_point = int(len(features) * 8/ 10)
    train_x, train_y = features[:split_train_point], labels[:split_train_point]
    val_x, val_y = features[split_train_point:], labels[split_train_point:]
    return train_x,train_y,val_x,val_y

def process_label(data):
    new_labels = []
    for i in data:
        label = i.squeeze()
        new_labels.append(label)
    return new_labels

def normalize(train_x, val_x):
    train_mean = np.average(train_x)
    val_mean = np.average(val_x)
    train_std = np.std(train_x)
    val_std = np.std(val_x)
    train_x /= 6
    val_x /= 6
    train_x = (train_x - train_mean) / train_std
    
    val_x  = (val_x - val_mean) / val_std
    return train_x, val_x


def load():
    train_x,train_y,val_x,val_y = load_dataset()
    
    train_y = np.array(process_label(train_y))
    val_y = np.array(process_label(val_y))
    train_x, val_x = normalize(train_x, val_x)
    train_x = train_x.astype(np.float32)
    val_x = val_x.astype(np.float32)
    train_y = train_y.astype(np.int32)
    val_y = val_y.astype(np.int32)
    return train_x,train_y,val_x,val_y