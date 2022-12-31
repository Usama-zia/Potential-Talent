"""Train similarity models."""

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

class similarity_models():
    #jaccard similarity
    def Jaccard_Similarity(doc1, doc2):
        # List the unique words in a document
        words_doc1 = set(doc1.lower().split())
        words_doc2 = set(doc2.lower().split())

        # Find the intersection of words list of doc1 & doc2
        intersection = words_doc1.intersection(words_doc2)

        # Find the union of words list of doc1 & doc2
        union = words_doc1.union(words_doc2)

        # Calculate Jaccard similarity score
        # using length of intersection set divided by length of union set
        return float(len(intersection)) / len(union)