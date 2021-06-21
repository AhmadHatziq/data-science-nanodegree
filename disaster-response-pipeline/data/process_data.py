'''
    The purpose of this python file is to do the ETL process ie 
    combine / merge the data in disaster_categories.csv and disaster_messages.csv
    into a single MESSAGES.db file. 
    
    Sample usage: 
    python process_data.py disaster_messages.csv disaster_categories.csv MESSAGES.db
    
'''

import sys

import numpy as np 
import pandas as pd

from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Loads the disaster categories and messages data from their respective csvs, merges them and
    returns a single combined dataframe. 
    
    Args:
        param1 (messages_filepath): The filepath to disaster_messages.csv.
        param2 (categories_filepath): The filepath to disaster_categories.csv.
        
    Returns: 
        output1 (combined_df): Dataframe containing messages and categories data. 
    '''

    # Load both csv files. 
    categories_df = pd.read_csv(categories_filepath)
    messages_df = pd.read_csv(messages_filepath)
    
    # Merge both dataframes. 
    combined_df = messages_df.merge(categories_df, on = ['id'])
    
    return combined_df


def clean_data(combined_df):
    '''
    Cleans the dataframe of combined messages and categories data. 
    
    Args:
        param1 (combined_df): The dataframe of merged messages and categories data. 
        
    Returns: 
        output1 (combined_df): Cleaned dataframe.
    '''
    
    # Split categories into seperate category columns. 
    categories = combined_df['categories'].str.split(';', expand = True)
    
    # Use first row to extract a list of new column names for categories. 
    row = categories.iloc[0]
    category_colnames = row.transform(lambda x: x[:-2]).tolist() 
    categories.columns = category_colnames
    
    # Convert category values to binary values.
    for column in categories: 
        # set each value to be the last character of the string.
        categories[column] = categories[column].transform(lambda x: x[-1:])
        
        # convert column from string to numeric.
        categories[column] = pd.to_numeric(categories[column])
        
    # Replace categories column in combined_df with new category columns. 
    combined_df.drop('categories', axis = 1, inplace = True)
    combined_df = pd.concat([combined_df, categories], axis = 1)
    
    # Remove duplicates. 
    combined_df.drop_duplicates(inplace = True)
    
    # Convert 'related' column to binary. Set all '2' to '1'.  
    combined_df['related'].loc[(combined_df['related'] == 2)] = 1
    
    # Log combined_df.head(). 
    print('Sample data from cleaned dataframe:')
    print(combined_df.head())
    
    return combined_df

def save_data(combined_df, database_filename):
    '''
    Saves the input dataframe into a sqlite db file. 
    
    Args:
        param1 (combined_df): The dataframe to be saved. 
        param2 (database_filename): The db file name. 
        
    Returns: 
        Does not return anything. 
    '''
    
    create_engine_string = r"sqlite:///" + database_filename
    engine = create_engine(create_engine_string)
    combined_df.to_sql('MESSAGES', engine, index = False, if_exists = 'replace')
    return


def main():
    '''
    Main driver function for the ETL process. 
    '''

    print('Starting ETL process...')

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        print(messages_filepath, categories_filepath, database_filepath)
        
        
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        
        print('Cleaning data...')
        df = clean_data(df)
        
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
        
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()