import pandas as pd
df=pd.read_csv('path of the file/ElevateEnergy.csv',encoding='latin')
df[:5]
newdf=df[['text','created_at']]
import re
import string
string.punctuation
#removing url from text
def remove_mentionurls(text):
    text_out=re.sub(r'@[A-Za-z0-9]+','',text)
    re.sub('https?://[A-za-z0-9]+','',text)
    return text_out
def remove_nonalphanumeric(text):
    text_out="".join([char for char in text if char not in string.punctuation])
    return text_out

newdf['cleaned_text']=newdf['text'].apply(lambda x: remove_mentionurls(x))
newdf['cleaned_text']=newdf['text'].apply(lambda x: remove_nonalphanumeric(x))

from nltk.corpus import stopwords
def remove_stopwords(text):
    stopwords_list = stopwords.words('english')
    whitelist = ["n't", "not", "no"]
    words = text.split()
    clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1]
    return " ".join(clean_words)
newdf['remove_stopword']=newdf['cleaned_text'].apply(lambda x: remove_stopwords(x))

from nltk.stem import PorterStemmer
def stemming(text):
    porter = PorterStemmer()
    words = text.split()
    stemmed_words = [porter.stem(word) for word in words]
    return " ".join(stemmed_words)

newdf['stemmed_words']=newdf['remove_stopword'].apply(lambda x: stemming(x))

def tokenization(text):
    tokens=re.split('\W+',text)
    return tokens
newdf['tokens']=newdf['stemmed_words'].apply(lambda x: tokenization(x))
  
from collections import Counter 
fdist=Counter(" ".join(newdf["stemmed_words"]).split()).most_common(15)
fdist
from nltk.probability import FreqDist
fdist = FreqDist(newdf['tokens'])
print(fdist
