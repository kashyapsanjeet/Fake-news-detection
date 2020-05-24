#Make Necessary imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df=pd.read_csv('news.csv')

#removing unwanted rows
df.drop('Unnamed: 0',axis=1,inplace=True)

#if want to use Count vectorizer
#import string

#def text_process(mess):
   # """
   # Takes in a string of text, then performs the following:
   # 1. Remove all punctuation
   # 2. Remove all stopwords
   # 3. Returns a list of the cleaned text
   # """
   # # Check characters to see if they are in punctuation
   # nopunc = [char for char in mess if char not in string.punctuation]

   # # Join the characters again to form the string.
   # nopunc = ''.join(nopunc)
    
   # # Now just remove any stopwords
   # return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
   
#raw documents into a matrix of TF-IDF features
Tfidf=TfidfVectorizer(stop_words='english',max_df=0.3)

#splitting Data For testing
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=0)

#transforming Data
tfidf_x_train=Tfidf.fit_transform(X_train)

#transtoming test data
tfidf_x_test=Tfidf.transform(X_test)

#initializing classifier
pac=PassiveAggressiveClassifier(C=1,max_iter=10)
pac.fit(tfidf_x_train,y_train)

y_pred=pac.predict(tfidf_x_test)

score=accuracy_score(y_test,y_pred)
print('Accuracy without gridSearchCV',score)

#finding the best parameters for PassiveAgressiveClassifier
from sklearn.model_selection import GridSearchCV
param={'C':[1,0.1,10,100,1000,10000],'max_iter':[1,10,20,50,100,200]}
grid=GridSearchCV(PassiveAggressiveClassifier(),param)
grid.fit(tfidf_x_train,y_train)

pac=PassiveAggressiveClassifier(C=10,max_iter=20)
pac.fit(tfidf_x_train,y_train)
pred3=pac.predict(tfidf_x_test)
score=accuracy_score(y_test,pred3)
print('Accuracy_percent={}'.format(round(score*100,2)))


