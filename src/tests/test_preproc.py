import sys
import pandas as pd
sys.path.append("../modules")

from preprocess import PreprocessAPA

obj = PreprocessAPA()
interview = "/Users/ibragimzhussup/Desktop/APA_Lab/src/data/interview.jsonl"
letters = "/Users/ibragimzhussup/Desktop/APA_Lab/src/data/leserbrief.jsonl"
comments = "/Users/ibragimzhussup/Desktop/APA_Lab/src/data/meinung.jsonl"

data = obj.preprocess_data(interview, letters, comments)
print(data.head())

full = obj.preprocess_text(data)
print(data.head())

sample = full.iloc[:10]['text']
for i,s in enumerate(sample):
    print(f"Text number {i}\n{s}\n")
    
full_ml = obj.ml_text_preproc(full, text_column="text", label_column="labels", full_preproc=False)
prep_ml = obj.ml_text_preproc(full, text_column="text", label_column="labels", full_preproc=True)

for i, t in enumerate(full_ml['text'].iloc[:2]):
    print(f"Text number {i}\n{t}\n")
    
for i, t in enumerate(prep_ml['text'].iloc[:2]):
    print(f"Text number {i}\n{t}\n")
    