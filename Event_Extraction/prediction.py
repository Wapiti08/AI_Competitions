import pandas as pd
import plac
import random
import warnings
import pickle
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding

def position_inverse(position_tuple, context):
    substring = context[position_tuple[0]:position_tuple[1]]
    return substring


def prediction_entity(df, output_dir = None):
    # test the saved model
    print("Loading from", output_dir)
    nlp = spacy.load(output_dir)
    labels = ['trigger', 'object', 'subject', 'time', 'location']
    test_dict = {'id':df['id'].tolist(),
                'trigger':[''],
                 'object':[''], 
                 'subject':[''], 
                 'time':[''], 
                 'location':['']}
            
    df_test = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in test_dict.items() ]))

    for index in range(len(df['news'])):
        text = df['news'][index]
        for ent in nlp(text).ents:
            for label in labels:
                if label == ent.label_:
                    df_test[label][index] = position_inverse((ent.start_char, ent.end_char), ent.text)
    return df_test

# load the testing data
df = pd.read_csv('./Dataset/test.csv')
# load the context 

df_test = prediction_entity(df, output_dir = './model')


# save the result
df_test.to_csv('./Dataset/test_prediction_100.csv',index=False)
