from BuildModels import BuildModels
from Neural_Nets import RNN_LSTM, CNN_LSTM, GRU_LSTM

# running the different models
model = BuildModels()

# Supervised Learning
model.naive_bayes()
model.logistic_regression()
model.svm()
model.random_forest()
model.decision_tree()

# Unsupervised Learning
# model.kmeans()

# Neural Networks
RNN_LSTM()
CNN_LSTM()
GRU_LSTM()