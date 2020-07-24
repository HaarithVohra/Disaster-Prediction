import pandas as pd
from sklearn import model_selection

from Preprocessing import Preprocessing

data = pd.read_csv("disaster_prediction.csv", encoding="utf-8")

# save data that doesn't have confidence on one, to predict later
data_unconfident = data[data.confidence != 1]
data_unconfident.to_csv("../Data/data_unconfident.csv", index=False)

data.drop("choose_one_gold", axis=1, inplace=True) # removing columns we don't need
data.drop(data[data.choose_one == "Can't Decide"].index, inplace=True) # delete rows we don't need
data.drop(data[data.confidence != 1].index, inplace=True) # delete rows with confidence != 1
data.drop(data[data.choose_one == "confidence"].index, inplace=True) # delete columns we don't need

data.reset_index(inplace=True, drop=True) # reset the indices

# label the remaining data
data.choose_one[data.choose_one == "Not Relevant"] = 0
data.choose_one[data.choose_one == "Relevant"] = 1

# split the data into training and test
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(data["text"], data["choose_one"], test_size=0.2)

# reset the indices for the Series and convert them to dataframes
Train_X.reset_index(inplace=True, drop=True)
Train_Y.reset_index(inplace=True, drop=True)

training_data = pd.concat([Train_X, Train_Y], axis=1) # getting the training data

Test_X.reset_index(inplace=True, drop=True)
Test_Y.reset_index(inplace=True, drop=True)

testing_data = pd.concat([Test_X, Test_Y], axis=1) # getting the testing data

# Preprocessing the training data
preprocessed_train = Preprocessing(training_data)
preprocessed_train.to_csv_file("preprocessed_training_data") # saves the file in the ../Data directory

# Preprocessing the testing data
preprocessed_test = Preprocessing(testing_data)
preprocessed_test.to_csv_file("preprocessed_testing_data") # saves the file in the ../Data directory

train = pd.read_csv("../Data/preprocessed_training_data.csv", encoding="utf-8")
test = pd.read_csv("../Data/preprocessed_testing_data.csv", encoding="utf-8")