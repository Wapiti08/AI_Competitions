import plac
import random
from pathlib import Path
import pandas as pd
import pickle
import spacy
import warnings
from spacy.util import minibatch, compounding


def entity_create(context, substring_list, labels):
    # build the entity_list
    entity_list = []
    for substring, label in zip(substring_list, labels):
        # return the start index
        # print(substring)
        try:
            start_index = context.find(substring)
            # return the end index
            end_index = start_index + len(substring)
            entity_list.append((start_index, end_index, label))
        except:
            print("There is a Nan value here .. Ignore")
            continue
    # check the overlap
    
    return entity_list

def train_create(df, labels):
    # build the train_data list
    train_data = []
    for i in range(len(df)):
        # get the context
        context = df.iloc[i,:]['news']
        # if i >= 1:
        #     # process the same context
        #     if context == df.iloc[i-1,:]['news']:
                
        #     else:
        #         continue

        # get the substring
        substring_list = df.iloc[i,:][labels]
        # print(substring_list)
        entity_list = entity_create(context, substring_list, labels)
        train_data.append((context, {"entities": entity_list} ))
    return train_data

# read the training csv
df = pd.read_csv('./Dataset/train.csv')
# fill the Nan with 0
labels = df.columns.values[2:7]

train_data = train_create(df, labels)

# save the train_data to a pickle file
with open('./Dataset/train_data.pkl', 'wb') as f:
    pickle.dump(train_data, f)


