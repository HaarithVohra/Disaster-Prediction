import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import path
from PIL import Image

import wordcloud_make

# this data frame contains the preprocessed data
data = pd.read_csv("../preprocessed_data.csv", sep=",", encoding="utf-8")
data_orig = pd.read_csv("../Preprocessing/disaster_prediction.csv", sep=",", encoding="utf-8")

# removing columns we don't need
data_orig.drop("choose_one:confidence", axis=1, inplace=True)
data_orig.drop("choose_one_gold", axis=1, inplace=True)

# delete rows we don't need
data_orig.drop(data_orig[data_orig.choose_one == "Can't Decide"].index, inplace=True)

# reset the indices
data_orig.reset_index(inplace=True, drop=True)

# label the remaining data
data_orig.choose_one[data_orig.choose_one == "Not Relevant"] = 0
data_orig.choose_one[data_orig.choose_one == "Relevant"] = 1

dataRelevant_orig = [data_orig.text[i] for i in range(len(data_orig)) if data_orig.choose_one[i] == 1]
dataIrrelevant_orig = [data_orig.text[i] for i in range(len(data_orig)) if data_orig.choose_one[i] == 0]

dataRelevant = [data.text[i] for i in range(len(data)) if data.choose_one[i] == 1]
dataIrrelevant = [data.text[i] for i in range(len(data)) if data.choose_one[i] == 0]

# compose all relevant tweets into a string
relevantText = " ".join(tweet for tweet in dataRelevant)
# compose all irrelevant tweets into a string
irrelevantText = " ".join(tweet for tweet in dataIrrelevant)
# all data
all_data = " ".join(data.text)

# compose all relevant tweets into a string
relevantText_orig = " ".join(tweet for tweet in dataRelevant_orig)
# compose all irrelevant tweets into a string
irrelevantText_orig = " ".join(tweet for tweet in dataIrrelevant_orig)
# all data
all_data_orig = " ".join(data_orig.text)

wordcloud_make.make_wordcloud(relevantText, "relevant")
wordcloud_make.make_wordcloud(irrelevantText, "irrelevant")
wordcloud_make.make_wordcloud(all_data, "all")

wordcloud_make.make_wordcloud(relevantText_orig, "relevant_orig")
wordcloud_make.make_wordcloud(irrelevantText_orig, "irrelevant_orig")
wordcloud_make.make_wordcloud(all_data_orig, "all_orig")
