import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

class PreprocessAPA:
    
    # Initialize the paths
    def __init__(self):
        self.stop_words = set(stopwords.words('german'))
        self.stemmer = SnowballStemmer("german")
    
    def preprocess_data(self, interview: str, letters: str, comments: str):
        
        interview = pd.read_json(open(interview), lines=True)
        letters = pd.read_json(open(letters), lines=True)
        comments = pd.read_json(open(comments), lines=True)
        
        data = [interview, letters, comments]

        # Extract data from list values in "labels" column
        for i in range(len(data)):
            data[i]['labels'] = data[i]['labels'].apply(lambda x: x[0])
            
        interview = data[0]
        letters = data[1]
        comments = data[2]
        
        # Label if relevant, else: NON-RELEVANT
        interview['labels'] = interview['labels'].apply(lambda x: "interview" if x == "RELEVANT" else x)
        letters['labels'] = letters['labels'].apply(lambda x: "letter" if x == "RELEVANT" else x)
        comments['labels'] = comments['labels'].apply(lambda x: "comment" if x == "RELEVANT" else x)
        
        # Concatnate the data
        data = pd.concat([interview, letters, comments], ignore_index=True).drop_duplicates(subset=['labels', 'text'])
        
        
        # HANDLE DUPLICATED LABELS
        print("Handling duplicates")
        duplicates = data[data.duplicated(subset='text', keep=False)][['labels', 'text']].groupby('text')['labels'].apply(list).reset_index()
        duplicates['len'] = duplicates['labels'].apply(lambda x: len(x))
        tripple_duplicates = duplicates[duplicates['len'] == 3]
        if len(tripple_duplicates) >= 100:
            print(f"WARNING: deleting {len(tripple_duplicates)} rows")
        data = data[~data['text'].isin(tripple_duplicates['text'])]
        double_duplicates = duplicates[duplicates['len']==2]
        to_delete = data[data['text'].isin(double_duplicates['text'])][data['labels']=='NONRELEVANT']
        to_delete_indicies = to_delete.index
        data = data.drop(to_delete_indicies)
        if len(data[data.duplicated(subset="text")]) == 0:
            print("Deduplicated successfully")
            print("Data preprocessed successfully")
        else:
            print("Deduplication not successful")
            print(data[data.duplicated(subset="text")])
            
        return data
        

    def preprocess_text(self, data):
        
        words = ['leserpost', 'leserbrief']
        def word_remover(text):
            text = text.lower()
            for word in words:
                text = text.replace(word,'')
            return text
        
        data = data.copy()
        data['text'] = data['text'].apply(word_remover) # Remove all unnecessary words
        data['text'] = data['text'].apply(lambda text: re.sub(r'[^\w\s\d]', '', text)) # Replace all punctuations except, spaces, words and numbers
        data['text'] = data['text'].apply(lambda text: re.sub(r'\s+', ' ', text))  # Replace multiple whitespaces with a single space
        data['text'] = data['text'].apply(lambda text: text.strip()) # Strip all text in each row
        return data

    
    def ml_text_preproc(self, data, text_column: str, label_column: str, full_preproc: bool):
        
        def preprocess_text(text):
            # Remove URLs
            text = re.sub(r'http\S+', '', text)
            # Tokenization, lowercasing, and removing stopwords
            words = word_tokenize(text.lower())
            cleaned_words = [word for word in words if word.isalnum() and word.lower() not in self.stop_words]
            return ' '.join(cleaned_words)
        
        df = data.copy()
        if full_preproc:
            df[text_column] = df[text_column].apply(preprocess_text)
            
        if df[label_column].dtype == "O":
            df["label_ids"] = df[label_column].factorize()[0]
            
        return df
    
    def split_data(self, data, test_val: bool, n_samples_train: int, test_split_ratio = 0.5):
        
        """
        Example usage:
        
        if splitted into train,test and validation:
        
        train, test, val = obj.split_data(data, test_val=True, n_samples_train=500, test_split_ratio=0.5)
        
        train, test = obj.split_data(data, test_val=False, n_samples_train=500)
        
        Returns: balanced trainset and testset as they are.
        
        """
        
        balanced_train_set = pd.DataFrame()
        for label_id in data['label_ids'].unique():
            label_data = data[data['label_ids'] == label_id]
            sampled_data = label_data.sample(n=n_samples_train, random_state=42)
            balanced_train_set = pd.concat([balanced_train_set, sampled_data], ignore_index=True)

        # Create the test set with the remaining data
        test_set = data.drop(balanced_train_set.index)

        # Shuffle the balanced training set
        balanced_train_set = shuffle(balanced_train_set, random_state=42)
        
        if test_val:
            testset, valset = train_test_split(test_set, test_size=test_split_ratio)
            return balanced_train_set, testset, valset
        else:
            return balanced_train_set, test_set
            
            