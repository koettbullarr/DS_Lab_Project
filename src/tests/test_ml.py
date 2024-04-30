import sys
sys.path.append("/Users/ibragimzhussup/Desktop/APA_Lab/src/modules")

from preprocess import PreprocessAPA
from ML_models import ML_Models
import pandas as pd
import matplotlib.pyplot as plt

prep_ml = pd.read_csv("/Users/ibragimzhussup/Desktop/APA_Lab/src/data/preprocessed_text.csv")

ml_modeling = ML_Models()
preproc = PreprocessAPA()
train, test, val = preproc.split_data(prep_ml, test_val=True, n_samples_train=500, test_split_ratio=0.5)

train[['labels', 'label_ids']].drop_duplicates().sort_values(by='label_ids')

preprocessed_df_performance = ml_modeling.run(train, test)

ml_modeling.plot_performance(preprocessed_df_performance, average_val = "both")
plt.show()

ml_modeling.plot_performance(preprocessed_df_performance, average_val = "weighted")
plt.show()

ml_modeling.plot_performance(preprocessed_df_performance, average_val = "macro")
plt.show()

lr_bestparams = ml_modeling.fine_tune("logreg", use_all_CPUs=True, number_of_iterations=50, num_cv=5)
svc_bestparams = ml_modeling.fine_tune("svc", use_all_CPUs=True, number_of_iterations=10, num_cv=5)


from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

lr_finetuned = LogisticRegression(**lr_bestparams)
svc_finetuned = SVC(**svc_bestparams)

X_train, y_train = train['text'], train['label_ids']
X_test, y_test = test['text'], test['label_ids']
X_val, y_val = val['text'], val['label_ids']

vect = TfidfVectorizer()
X_train_vect = vect.fit_transform(X_train)
X_test_vect = vect.transform(X_test)
X_val_vect = vect.transform(X_val)

lr_finetuned.fit(X_train_vect, y_train)
svc_finetuned.fit(X_train_vect, y_train)

print("Test classification report")
print("Logreg")
print(classification_report(y_test, lr_finetuned.predict(X_test_vect)))
print("SVC")
print(classification_report(y_test, svc_finetuned.predict(X_test_vect)))


print("Validation classification report")
print("Logreg")
print(classification_report(y_val, lr_finetuned.predict(X_val_vect)))
print("SVC")
print(classification_report(y_val, svc_finetuned.predict(X_val_vect)))


# Step 1: Create Binary Classifiers
classifiers = {}


for label in [0, 1, 2, 3]:
    # Create a binary classifier for the current label
    clf = LogisticRegression()
    # Define binary labels (1 for the current label, 0 for others)
    y_train_binary = (y_train == label).astype(int)
    # Train the binary classifier
    clf.fit(X_train_vect, y_train_binary)
    # Store the trained classifier
    classifiers[label] = clf

# Step 2: Make Predictions
predictions_proba = {}

for label, clf in classifiers.items():
    # Predict probabilities for the current label
    predictions_proba[label] = clf.predict_proba(X_test_vect)[:, 1]


import numpy as np
# Step 3: Combine Predictions
combined_predictions = np.argmax(np.array(list(predictions_proba.values())), axis=0)

print(classification_report(y_test, combined_predictions))