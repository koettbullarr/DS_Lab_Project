import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("/Users/ibragimzhussup/Desktop/APA_Lab/src/modules")

from preprocess import PreprocessAPA
full_ml = pd.read_csv("/Users/ibragimzhussup/Desktop/APA_Lab/src/data/full_text.csv")

preproc = PreprocessAPA()
train, test, val = preproc.split_data(full_ml, test_val=True, n_samples_train=500, test_split_ratio=0.5)
print(train[['labels', 'label_ids']].drop_duplicates().sort_values(by='label_ids'))

from DL_models import RNN, CNN 


lstm = RNN()
lstm.tokenize(train,test,val)
X_train, X_test, X_val, y_train, y_test, y_val = lstm.pad_and_label_preproc()
model_lstm, _ = lstm.build_model(use_basic_embed=True, optimizer="sgd")
print(model_lstm.summary())

cnn = CNN()
cnn.tokenize(train,test,val)
X_train, X_test, X_val, y_train, y_test, y_val = cnn.pad_and_label_preproc()
model_cnn, _ = cnn.build_model(use_basic_embed=True, optimizer="adam")
print(model_cnn.summary())


