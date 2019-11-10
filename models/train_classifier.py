import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


def load_data(database_filepath):
    """To load data from database and put the data into DataFrame
    Args:
        database_filepath(str): path of SQLite database
    Return:
        X(array-like): array of message texts
        Y(DataFrame): DataFrame of multiple target variables
        category_names(array): labels for categories
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql(con=engine,sql='messages')
    X = df['message']
    Y = df.drop(['id','message','original','genre'],axis=1)
    
    # drop columns where there is no records
    Y = Y.drop(Y.columns[Y.sum(axis=0)==0],axis=1)

    category_names = Y.columns.values

    return X, Y, category_names

def tokenize(text):
    """Tokenize series of texts and build bag of words as DataFrame.
    Args:
        text(array-like): array of text messages
    Return:
        text_counts(DataFrame): bag of words
    """
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    cv = CountVectorizer(
        lowercase=True, stop_words='english', ngram_range = (1,1), tokenizer = token.tokenize)
    text_counts = cv.fit_transform(text)
    return text_counts    

class TermNormalizer():
    def fit(self, X_train, Y_train):
        return self

    def transform(self, X_train):
        """
            Reference : https://medium.com/@datamonsters/text-preprocessing-in-python-steps-tools-and-examples-bf025f872908
            """
        ps = PorterStemmer()
        # Omit numbers, convert to lowercase, and then normalize terms
        transformed = list(
            map(lambda text:' '.join(map(lambda word: ps.stem(word), text.split())), 
                map(lambda text: text.lower(), 
                    map(lambda text: re.sub(r'\d+', '', text), X_train)))) 

        return transformed

def build_model():
    """To build ML model instance. Since there are mutliple target variables,
        to use MultiOutputClassifier.
    Args:
        None
    Return:
        model(MultiOutputClassifier):ML model instance generated.
    """
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    model = Pipeline([
         ('norm',TermNormalizer())
        ,('vect', CountVectorizer(lowercase=True, stop_words='english', tokenizer = token.tokenize))
        ,('tfidf',TfidfTransformer())
        ,('clf', MultiOutputClassifier(DecisionTreeClassifier()))
        ])
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """To print evaluation result of ML model.
    Args:
        model(MultiOutputClassifier): fitted instance of ML model
        X_test(DataFrame): test data of explanatory variables
        Y_test(DataFrame): test data of target variables
    Return:
        None
    """
    Y_test_pred = model.predict(X_test)
    report = classification_report(Y_test['request'], Y_test_pred[:,1], labels=[1])


def save_model(model, model_filepath):
    """To save ML model into a pickle file.
    Args:
        model(MultiOutputClassifier): ML model instance
        model_filepath: file path for saving model pickle file
    Return:
        None
    """
    pickle.dump(model,open(model_filepath,'wb+'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
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

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
