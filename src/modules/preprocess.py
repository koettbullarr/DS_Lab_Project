import pandas as pd
import string
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

class PreprocessAPA:
    
    # Initialize the paths
    def __init__(self, interview_path, letters_path, comments_path):
        self.interview_path = interview_path
        self.letters_path = letters_path
        self.comments_path = comments_path
        self.stop_words = set(stopwords.words('german'))
        self.stemmer = SnowballStemmer("german")
    
    # Load the data with pandas
    def load_data(self):
        interview = pd.read_json(self.interview_path, lines=True)
        letters = pd.read_json(self.letters_path, lines=True)
        comments = pd.read_json(self.comments_path, lines=True)
        return interview, letters, comments
    
    # Extract the labels because they're lists
    def extract_labels(self, data):
        for i in range(len(data)):
            data[i]['labels'] = data[i]['labels'].apply(lambda x: x[0])
        return data
    
    
    def filter_relevant(self, data):
        
        """
        THIS FUNCTION FILTERS DATA TO INCLUDE ONLY RELEVANT DATA

        Returns:
            _type_: _pd.DataFrame_
        """
        
        for i in range(len(data)):
            data[i] = data[i][data[i]['labels'] == 'RELEVANT']
        return data
    
    
    def rewrite_labels(self, data):
        
        """
        THIS FUNCTION CAN ONLY BE USED AFTER USAGE OF filter_relevant()
        REWRITES THE LABELS

        Returns:
            _type_: _pd.DataFrame_
        """
        
        for i in range(len(data)):
            if i == 0:
                data[i].loc[:, 'labels'] = 'interview'
            elif i == 1:
                data[i].loc[:, 'labels'] = 'letter'
            else:
                data[i].loc[:, 'labels'] = 'comment'
        return data
    
    # Concats the data
    def concat_data(self, data):
        return pd.concat(data, ignore_index=True)
    
    def preprocess_text(self, df):
        
        """
        DELETES WORDS "LESERPOST" AND "LESERBRIEF"

        Returns:
            _type_: _pd.DataFrame_
        """
        words = ['leserpost', 'leserbrief']
        def word_remover(text):
            text = text.lower()
            for word in words:
                text = text.replace(word,'')
            return text

        df['text'] = df['text'].apply(word_remover)
        return df
    
    # Removes punctuations
    def remove_punctuation(self, df, text_column: str):
        def remove_special_chars(text):
            translator = str.maketrans('', '', string.punctuation)
            return text.translate(translator)
        
        df.loc[:, text_column] = df[text_column].apply(remove_special_chars)
        return df
    

    # Replace whitespaces with " "
    def remove_whitespace(self, df, text_column: str):
        """
        Replaces all chars like newline or tabline with space

        Args:
            df (_type_): _description_
            text_column (str): _description_
        """
        def clean_whitespaces(text):
            translator = str.maketrans(string.whitespace, ' ' * len(string.whitespace))
            return text.translate(translator)
        
        def reduce_spaces(text):
            return re.sub(r'\s+', ' ', text)
        
        df.loc[:, text_column] = df[text_column].apply(clean_whitespaces)
        df.loc[:, text_column] = df[text_column].apply(reduce_spaces)
        
        return df
    
    def ml_text_preproc(self, df, text_column: str, 
                        label_column: str, full_preproc: bool):
        
        """
        PREPROCESSES TEXT FOR MACHINE LEARNING
        
        1) CREATES LABEL IDS IF LABEL == TYPE OBJECT
        2) IF FULL PREPROC ACTIVATED, DOES FULL ML PREPROC
        """
        def first_preproc(text):
            # Remove URLs
            text = re.sub(r'http\S+', '', text)
            # Tokenization and lowercasing
            words = word_tokenize(text.lower())
            # Removing stopwords and Stemming
            #REMOVED self.stemmer.stem(word) AND REPLACED WITH word
            cleaned_words = [word for word in words if word.isalnum() and word.lower() not in self.stop_words]
            return ' '.join(cleaned_words)
        
        if full_preproc:
            df.loc[:, text_column] = df[text_column].apply(first_preproc)
            
        if df[label_column].dtype == "O":
            df.loc[:, "label_ids"] = df[label_column].factorize()[0]
            
        return df
            
            