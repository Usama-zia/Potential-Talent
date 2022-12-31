"""find embeddings for text dataset."""

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

import warnings

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import scipy.stats as stats

import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine


import shutil
import re
import string
import spacy
import os
import wget

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import gensim
import gensim.downloader as api
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from gensim.models.doc2vec import Doc2Vec

from sentence_transformers import SentenceTransformer

class embeddings():
    def get_glove_embeddings(data,glove_model):
        """This function will be used to find the the GloVe Vectors for Job titles.

        args:

        data (list): list of job titles

        glove_model (dictionary): GloVe pre trained embeddings

        return:

        glove_vectors (list) : list of GloVe Vectors for job titles

        """

        glove_vectors = []
        token_vector = []
        #This loop will go through every job title, tokenize it,
        #and then find if the words are in glove model
        #and then find embeddings for those words
        for title in data:
            token_vector = []
            sentence_tokens = word_tokenize(title)
            words = [word for word in sentence_tokens if word in glove_model.index_to_key]
            for token in sentence_tokens:
                if token in words:
                    vector = glove_model[token]
                    token_vector.append(vector)
            glove_vectors.append(token_vector)

        return glove_vectors

    # Get Doc2Vec embeddings
    def get_doc2vec_embeddings(data, doc2vec_model):
        """This function will be used to find the the doc2vec Vectors for Job titles.

        args:

        data (list): list of job titles

        doc2vec_model (dictionary): Doc2Vec pre trained embeddings

        return:

        doc2vec_vectors (list) : list of doc2vec_vectors for job titles

        """
        doc2vec_vectors=[]
        for title in data:
            sentence_tokens = word_tokenize(title)
            tokens = list(filter(lambda x: x in doc2vec_model.wv.index_to_key, sentence_tokens))
            doc2vec_vectors.append(doc2vec_model.infer_vector(tokens))

        return doc2vec_vectors