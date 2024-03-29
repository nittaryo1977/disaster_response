import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """To load data from csv into DataFrame.
    Args:
        messages_filepath: Path to messages file
        categories_filepath: Path to categories file
    Return:
        df(DataFrame):DataFrame into which messages and categories are merged.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    """To clean data in DataFrame.
    Args:
        df(DataFrame):DataFrame before cleansing
    Return:
        df(DataFrame):DataFrame after cleansing
    """
     
    # Split categories into separate category columns
    categories = df['categories'].str.split(pat=';',expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[1,:]
    # use this row to extract a list of new column names for categories.
    category_colnames = categories.iloc[1,:].str.split('-',expand=True)[0]
    # rename the columns of `categories`
    categories.columns = category_colnames

    #Convert category values to just numbers 0 or 1.
    for column in categories:
	    # set each value to be the last character of the string
	    categories[column] = (categories[column]).str[-1:]
	
	    # convert column from string to numeric
	    categories[column] = categories[column].astype(int)
    
    # update 2 to 1 in column 'related'
    categories['related'][categories['related']==2] = 1
        
    # Replace categories column in df with new category columns
    df.drop('categories',axis=1,inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories],axis=1)
    # drop duplicates
    df = df[~df.duplicated(keep='first')]

    return df


def save_data(df, database_filename):
    """
    Args:
        df(DataFrame):cleansed DataFrame
        database_filename(str):file name for SQLite database
    Return:
    """
    engine = create_engine('sqlite:///' + database_filename)  
    df.to_sql('messages', engine, index=False, if_exists='replace')

    return


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

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
