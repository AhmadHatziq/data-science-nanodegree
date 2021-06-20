'''
    The purpose of this python file is to train a classifier on data extracted from the specified database and save the model to a pkl file. 
    
    Sample usage: 
    python train_classifier.py ../data/MESSAGES.db classifier.pkl
    
'''

import nltk
import numpy as np
import pandas as pd
import pickle
import re
import sys
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')
warnings.simplefilter('ignore')


"""
    Loads information from the database. 
    
    Args:
        param1 (database_filepath): Filepath to the database.
        
    Returns: 
        output1 (X): Columns to train on. 
        output2 (Y): Labels. 
        
"""
def load_data(database_filepath):
    # Create engine and connect to database. 
    create_engine_string = r"sqlite:///" + database_filepath
    engine = create_engine(create_engine_string)
    
    # Read in data from database
    df = pd.read_sql("SELECT * FROM MESSAGES", engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    
    return X, Y

"""
    Tokenizes a string text to a list of tokens. 
    
    Args:
        param1 (text): String of text.
        
    Returns: 
        output1 (normalized): List of tokens. 
        
    Sample: 
        Raw text: I am in Petionville. I need more information regarding 4636
        Tokenized: ['petionvil', 'need', 'inform', 'regard', '4636']
"""
def tokenize(text):

    # Converting everything to lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize words
    tokens = word_tokenize(text)
    
    # normalization word tokens and remove stop words
    normlizer = PorterStemmer()
    stop_words = stopwords.words("english")
    
    normlized = [normlizer.stem(word) for word in tokens if word not in stop_words]
    
    return normlized


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass

'''
    Main driver function for the model training process. 
'''
def main():

    print('Starting model training process...')

    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        
        print(X.head())
        print(Y.head())
        
        '''
        # Split to train and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
        '''

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/MESSAGES.db classifier.pkl')
    

if __name__ == '__main__':
    main()