from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize, MinMaxScaler, StandardScaler
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime
from keras.models import Sequential
from keras import layers


# encode the categorical data
class_le = LabelEncoder()

# ========== without issue_date and listing_date =============

# load trainig data
train_df = pd.read_csv('./Dataset/train.csv', usecols=['issue_date', 'listing_date', 'condition','color_type','length(m)','height(cm)','X1','X2','breed_category','pet_category'])
train_df['time_gap'] = (pd.to_datetime(train_df["listing_date"])-pd.to_datetime(train_df['issue_date'])).dt.days
# delete issue_date and listing_date
train_df = train_df[['condition','color_type','length(m)','height(cm)','X1','X2','time_gap','breed_category','pet_category']]
# train_df = pd.read_csv('./Dataset/train.csv', usecols=['condition','color_type','length(m)','height(cm)','X1','X2','breed_category','pet_category'])
train_x = train_df.iloc[:,:-2]
# ========== feature engineering =========
# replace nan with 2 first
train_x['condition'] = train_x['condition'].fillna(2)
# encode categorical value
train_x['color_type'] = class_le.fit_transform(train_x['color_type'])
train_x = StandardScaler().fit_transform(train_x)
print(train_x)
# get the training y
train_y_1 = train_df['breed_category']
train_y_2 = train_df['pet_category']

# load testing data
test_df = pd.read_csv('./Dataset/test.csv', usecols = ['issue_date', 'listing_date','condition','color_type','length(m)','height(cm)','X1','X2'])
test_df['time_gap'] = (pd.to_datetime(test_df["listing_date"])-pd.to_datetime(test_df['issue_date'])).dt.days
# delete issue_date and listing_date
test_df = test_df[['condition','color_type','length(m)','height(cm)','X1','X2','time_gap']]
# encode the second color_type data
# test_df = pd.read_csv('./Dataset/test.csv', usecols = ['condition','color_type','length(m)','height(cm)','X1','X2'])
test_df['condition'] = test_df['condition'].fillna(2)
test_df['color_type'] = class_le.fit_transform(test_df['color_type'])
test_x = StandardScaler().fit_transform(test_df)

# ====== without preprocessing ======
# use Logistic Regression to fit the breed_category

clf_1 = LogisticRegression(random_state=0).fit(train_x, train_y_1)

# predict the y1
y_1_pred = clf_1.predict(test_x)
print(y_1_pred)



# ========= Use MLP to fit the pet_category ===========
clf_2 = MLPClassifier(random_state=1, max_iter=300).fit(train_x, train_y_2)

# clf_1 = LogisticRegression(random_state=0).fit(train_x, train_y_1)

# predict the y1
y_2_pred = clf_2.predict(test_x)

# ======= use CNN to fit the pet_category ===========

# input_dim = train_x.shape[1]
# model = Sequential()
# model.add(layers.Dense(32, input_dim=input_dim, activation = 'relu'))
# model.add(layers.Dense(10, input_dim=input_dim, activation = 'relu'))
# model.add(layers.Dense(1, activation='softmax'))
# model.compile(loss='mse', metrics=['accuracy'], optimizer='adam')

# model.fit(train_x, train_y_2, epochs=10, batch_size=32)

# # predict the y2

# test_x = StandardScaler().fit_transform(test_x)
# # test_x = MinMaxScaler().fit_transform(test_x)
# y_2_pred = model.predict(test_x)
# print(y_2_pred)
# buid the prediction dictionary
test_pet_id = pd.read_csv('./Dataset/test.csv', usecols = ['pet_id'])

pred_dict = {'pet_id': test_pet_id['pet_id'].to_list(),
            'breed_category': y_1_pred[:],
            'pet_category': y_2_pred[:]}
pre_df = pd.DataFrame(pred_dict)
# save the dataframe to csv
pre_df.to_csv('./Dataset/prediction3.csv', index=False)

'''
Try with the time_gap:
0.3814

Try without the time_gap:
0.3814
'''
