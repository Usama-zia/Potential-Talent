"""Download Embeddings."""

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

class load_embeddings():
    """Download and prepare embeddings."""
    def load_pretrained_glove():
        #Checking if current directory already has trained model
        embeddings_exist = os.path.isfile('glove.42B.300d.txt.word2vec')
        if  embeddings_exist == False:
            url = 'http://nlp.stanford.edu/data/glove.42B.300d.zip'
            filename = wget.download(url)
            print(filename)
            shutil.unpack_archive('glove.42B.300d.zip')
            glove_filename='glove.42B.300d.txt'
            word2vec_output_file = glove_filename+'.word2vec'
            glove2word2vec(glove_filename, word2vec_output_file)

        glove_embeddings = KeyedVectors.load_word2vec_format('glove.42B.300d.txt.word2vec', binary=False)
        return glove_embeddings

    def doc2vec_embeddings():
        #Checking if current directory already has trained model
        doc2vec_model_exist = os.path.isfile('my_doc2vec_model')
        if  doc2vec_model_exist == False:
            dataset = api.load("text8")
            data = [d for d in dataset]
            def tagged_document(list_of_list_of_words):
                for i, list_of_words in enumerate(list_of_list_of_words):
                    yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])

            data_training = list(tagged_document(data))
            Doc2Vec_model  = gensim.models.doc2vec.Doc2Vec(vector_size=300, min_count=3, epochs=150)
            Doc2Vec_model.build_vocab(data_training)

            Doc2Vec_model.train(data_training, total_examples=Doc2Vec_model.corpus_count, epochs=Doc2Vec_model.epochs)
            fname = "my_doc2vec_model"
            Doc2Vec_model.save(fname)

        doc2vec = Doc2Vec.load('my_doc2vec_model')
        return doc2vec