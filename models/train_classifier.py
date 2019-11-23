import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('stopwords')

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


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
    """Tokenize text and return array of tokens
        Reference : https://medium.com/@datamonsters/text-preprocessing-in-python-steps-tools-and-examples-bf025f872908
    Args:
        text(array-like): array of text messages
    Return:
        list: list of tokenized tokens
    """
    reg_tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
    lem = WordNetLemmatizer()
    reg_tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
    lem = WordNetLemmatizer()
    lowered = text.lower() # convert all character to lower case
    number_removed = re.sub(r'\d+','',lowered) # remove numbers
    tokenized = reg_tokenizer.tokenize(number_removed) # tokenize sentence into words with RegExpTokenizer
    lemmatized = map(lambda token:lem.lemmatize(token), tokenized) # lemmatize
    stopwords_removed = [token for token in lemmatized if token not in stopwords.words('english')] # remove stopwords
    return list(stopwords_removed)

def build_model():
    """To build ML model instance. Since there are mutliple target variables,
        to use MultiOutputClassifier.
    Args:
        None
    Return:
        model(GridSearchCV):ML model instance generated.
    """
 
    pipeline = Pipeline([
        ('vect', CountVectorizer(lowercase=True, stop_words='english', tokenizer = tokenize))
        ,('tfidf',TfidfTransformer())
        ,('clf', MultiOutputClassifier(DecisionTreeClassifier()))
        ])
    parameters = {'clf__estimator__max_depth':[3,5,7]}
    model = GridSearchCV(pipeline, parameters, cv=5, iid=False)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """To print evaluation result of ML model.
    Args:
        model(GridSearchCV): fitted instance of ML model
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
