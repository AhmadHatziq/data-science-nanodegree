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

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sqlalchemy import create_engine

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
warnings.simplefilter('ignore')

def load_data(database_filepath):
    """
    Loads information from the database. 
    
    Args:
        param1 (database_filepath): Filepath to the database.
        
    Returns: 
        output1 (X): Columns to train on. 
        output2 (Y): Labels. 
        
    """

    # Create engine and connect to database. 
    create_engine_string = r"sqlite:///" + database_filepath
    engine = create_engine(create_engine_string)
    
    # Read in data from database
    df = pd.read_sql("SELECT * FROM MESSAGES", engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    
    return X, Y


def tokenize(text):
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

    # Converting everything to lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize words
    tokens = word_tokenize(text)
    
    # normalization word tokens and remove stop words
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")
    
    lemmatized = [lemmatizer.lemmatize(word).lower().strip() for word in tokens if word not in stop_words]
    
    return lemmatized
   
def get_metrics(y_test, y_pred):
    """
    Gets classification metrics (f_score, precision, recall).  
    
    Args:
        param1 (y_test): True labels. 
        param2 (y_pred): Predicted labels
        
    Returns: 
        output1 (results_df): Dataframe of classification metrics.  
        output2 (scores): List of averaged metrics. 
    """ 
    
    # Get dataframe of f_score, precision, recall for each category
    results_df = pd.DataFrame(columns=['Category', 'f_score', 'precision', 'recall'])
    num = 0
    for cat in y_test.columns:
        precision, recall, f_score, support = precision_recall_fscore_support(y_test[cat], y_pred[:,num], average='weighted')
        
        results_df.at[num + 1, 'Category'] = cat
        results_df.at[num + 1, 'f_score'] = f_score
        results_df.at[num + 1, 'precision'] = precision
        results_df.at[num + 1, 'recall'] = recall
        num += 1
        
    # Get averaged scores
    scores = {}
    scores['Average f_score'] =  results_df['f_score'].mean()
    scores['Average precision'] = results_df['precision'].mean()
    scores['Average recall'] = results_df['recall'].mean()

    return results_df, scores    

def build_model():
    """
    Returns a GridSearchCV object for training. 
    Best parameters are obtained from notebook.
    
    Returns: 
        output1 (GridSearchCV): GridSearchCV object. 
    """
    
    pipe = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
                    ])
                    
    parameters = {'clf__estimator__max_depth': [None],
              'clf__estimator__n_estimators': [50]}

    cv = GridSearchCV(pipe, parameters)   
    
    return cv


def evaluate_model(model, X_test, Y_test):
    """
    Generate predictions and returns classification metrics. 
    
    Returns: 
        output1 (results_df): Dataframe of classification metrics.  
        output2 (scores): List of averaged metrics. 
    """
    
    Y_preds = model.predict(X_test)
    return get_metrics(Y_test, Y_preds)


def save_model(model, model_filepath):
    """
    Saves the model to the specified file path.  
    
    Args: 
        input1 (model): Model to be saved.  
        input2 (model_filepath): File path to save the model in.
    """
    
    pickle.dump(model, open(model_filepath, 'wb'))
    return


def main():
    '''
    Main driver function for the model training process. 
    '''

    print('Starting model training process...')

    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        
        # Split to train and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        
        print('Training model...')
        model.fit(X_train, Y_train)
        print('Model training finished.')
        
        
        print('Evaluating model...')
        metrics_df, avg_metrics = evaluate_model(model, X_test, Y_test)
        print('Model performance:')
        print(avg_metrics)

        
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
        

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/MESSAGES.db classifier.pkl')
    

if __name__ == '__main__':
    main()