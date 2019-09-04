from nltk.tokenize import word_tokenize

from nltk.corpus import wordnet

from nltk.corpus import stopwords

from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.stem.wordnet import WordNetLemmatizer

import csv

import pandas
sr = stopwords.words('english')
df = pandas.read_csv('test.csv',encoding = 'windows-1252')
lmtzr = WordNetLemmatizer()	
for index,row in df.iterrows():
    mytext = df['views'][index]
    tokens = word_tokenize(mytext)
    for token in tokens:
        if token in sr:
	        tokens.remove(token)
    lemmatized = [lmtzr.lemmatize(word) for word in tokens]
  
    newtext=TreebankWordDetokenizer().detokenize(lemmatized)
    df.iat[index,2] = newtext		



print(df)		
df.to_csv('test.csv')







		
