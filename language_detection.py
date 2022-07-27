import pandas as pd
import string
import re
import warnings
warnings.filterwarnings('ignore')
import joblib

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('Language_detection.csv',error_bad_lines=False)

def removeSymbolsAndNumbers(text):        
        text = re.sub(r'[{}]'.format(string.punctuation), '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[@]', '', text)

        return text.lower()

def removeEnglishLetters(text):        
        text = re.sub(r'[a-zA-Z]+', '', text)
        return text.lower()


X0 = df.apply(lambda x: removeEnglishLetters(x.Text) if x.Language in ['Arabic']  else x.Text, axis = 1)
X1 = X0.apply(removeSymbolsAndNumbers)
y = df['Language']
x_train, x_test, y_train, y_test = train_test_split(X1,y, random_state=42)

vectorizer = TfidfVectorizer(ngram_range=(1,3), analyzer='char')

pipe = pipeline.Pipeline([
    ('vectorizer', vectorizer),
    ('clf', LogisticRegression())
])

pipe.fit(x_train,y_train)

joblib.dump(pipe,'pipe.joblib')
