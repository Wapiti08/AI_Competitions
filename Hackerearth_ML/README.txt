1. Decide which problem this question is: ---- classification
    (1) We can find the breed_category is kind of binary classification problem ---- Logistic Regression
    (2) pet_category is multi-classification --- original go with SVM

2. Which columns are not necessary and how to process categorical data:
    (1) pet_id is not necessary
    (2) issue_date and listing_date may not make a difference
    (3) Color needs to encode to numeric data (use LableEncoder() in sklearn.preprocessing instead of map function)

3. Try without issue_date and listing_date
    (1) In SVM, I tried with linear kernel first. The result is not well and all the prediction value is only one value
    (2) Then I tried with poly kernel, the result is still the same, which means the kernel is not the factor.
    (3) After that, I decided to use preprocessing to normalize the training and testing data, but the result is still the same.
        Obviously, the SVM is not the right choice.
    (4) Try with Decision Tree. It is still the same result.
    (5) Try with NN model --- MLP. This time the result is fine.
    (6) The result is not great. So I am thinking to change the preprocessing method. I tried with MinMaxScaler and StandardScaler as well.
    When I checked the result, I found there is not that prediction for pet_category with the value 4. 
    (7) Maybe the gap between issue_date and listing_date will make a difference. I decided to take this factor. But there is still no 4 value.
    And the accuracy was worse then before.
    (8) After I got some good feedback, I tried those two prediction will all NN model to see the performance.

4. When I did the test, I found there is Nan in column.
    (1) Then replace the Nan with 0 first. (Found nan only in condition column)
    (2) If the accuracy did not improve. I rechecked the columns again and used only height and length to see. The result is still the same.
    (3) When I recheck the condition for every pet_category 4. There are only nan or 2. So I replace the nan with 2 to see.
        df[df['pet_category']==4]['condition'].value_counts():
        2.0    103
        0.0     30
        1.0     25
        Name: condition, dtype: int64

5. Finally：
    (1) I included the time gap.
    (2) Filled the nan with 2.
    (3) Used LogisticRegression for binary-classification and MLP for multi-classification
    (4) Used the LableEncoder to encode color_type
    (5) Used the StandardScaler to normalize the data.
    (6) The prediction_cnn is the final code and the predicition3 is the final output.