from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# encode the categorical data
class_le = LabelEncoder()

# ========== without issue_date and listing_date =============

# load trainig data
train_df = pd.read_csv('./Dataset/train.csv', usecols=['condition','color_type','length(m)','height(cm)','X1','X2','breed_category','pet_category'])
train_x = train_df.iloc[:,:-2]
# encode the first color_type data
train_x = class_le.fit_transform(train_x.iloc[:,1:2])
train_y_1 = train_df.iloc[:,-2:-1]
train_y_2 = train_df.iloc[:,-1:]

# load testing data
test_df = pd.read_csv('./Dataset/test.csv', usecols = ['condition','color_type','length(m)','height(cm)','X1','X2'])
test_x = class_le.fit_transform(test_df.iloc[:,1:2])

# ====== without preprocessing ======
# use Logistic Regression to fit the breed_category
train_x = train_x.reshape(-1,1)
clf_1 = LogisticRegression(random_state=0).fit(train_x, train_y_1)

# predict the y1
test_x = test_x.reshape(-1,1)
y_1_pred = clf_1.predict(test_x)


# use SVM to fit the pet_category
# clf = SVC(kernel = 'poly')
clf = SVC(kernel = 'linear')
train_x = normalize(train_x, norm='l2')
clf.fit(train_x, train_y_2)

# predict the y2
test_x = normalize(test_x, norm='l2')
y_2_pred = clf.predict(test_x)

# buid the prediction dictionary
test_pet_id = pd.read_csv('./Dataset/test.csv', usecols = ['pet_id'])

pred_dict = {'pet_id': test_pet_id['pet_id'].to_list(),
            'breed_category': y_1_pred[:],
            'pet_category': y_2_pred[:]}
pre_df = pd.DataFrame(pred_dict)
# save the dataframe to csv
pre_df.to_csv('./Dataset/prediction.csv', index=False)