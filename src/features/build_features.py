"""Data preprocessing and prepration for analysis."""

import re
import string
import spacy

import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class text_processing():
    def preprocessing_data(data):
        print('============Before Preprocessing=============')
        for index,text in enumerate(data['job_title'][0:5]):
            print('job_title %d:\n'%(index+1),text)

        # Loading model
        #nlp = spacy.load('en_core_web_sm',disable=['parser', 'ner'])
        stop = stopwords.words('english')

        #removing punctuations
        data['job_title'] = data['job_title'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))
        data['connection'] = data['connection'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))
        data['connection'] = data['connection'].astype(str).astype(int)

        #replacing HR with Human Resources
        data['job_title'] = data['job_title'].apply(lambda x:x.replace('HR', ' Human Resources '))

        #removing extra spaces
        data['job_title'] = data['job_title'].apply(lambda x: re.sub(' +',' ',x))

        #lowercase the words
        data['job_title'] = data['job_title'].apply(lambda x: x.lower())

        #remove numbers
        data['job_title'] = data['job_title'].apply(lambda x: re.sub(r'\d+', '', x))

        #remove white spaces
        data['job_title'] = data['job_title'].apply(lambda x: x.strip())

        #Lemmatization with stopwords removal
        data['job_title']=data['job_title'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        #.apply(lambda x: ' '.join([token.lemma_ for token in list(nlp(x)) if (token.is_stop==False)]))

        #after Preprocessing
        print('============After Preprocessing=============')
        for index,text in enumerate(data['job_title'][0:5]):
            print('job_title %d:\n'%(index+1),text)

        return data

