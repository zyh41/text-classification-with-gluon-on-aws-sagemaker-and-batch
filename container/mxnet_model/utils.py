#!/usr/local/bin/python2
# -*- coding: utf-8 -*-
from string import punctuation, maketrans
import sklearn
import logging
import time
import math
import re
import itertools
from collections import Counter
from sklearn.model_selection import train_test_split
import pandas as pd
import sklearn
from sklearn.base import TransformerMixin
import os
import sys
import pickle
import cPickle
import boto3
from io import BytesIO
from io import StringIO
import json
import spacy
import mxnet.ndarray as nd
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from mxnet.gluon import nn, rnn

#---------------------------------------------------------------------
# config helpers
#---------------------------------------------------------------------

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def isint(value):
    try:
        int(value)
        return True
    except ValueError:
        return False  

def hasnumbers(value):
    return any(char.isdigit() for char in value)

def convert_config(config):
    # converts to proper format
    for k,v in config.items():
        # convert everyhthing to string from unicode
        v = config[k] = str(v)
        if isfloat(v):
            config[k] = float(v)
        if isint(v):
            config[k] = int(v)
        if v in {'none','None','null'}:
            config[k] = None
        if v in {'True', 'true'}:
            config[k] = True
        if v in {'false','False'}:
            config[k] = False

#---------------------------------------------------------------------
# download helpers
#---------------------------------------------------------------------

class DownloadManager(object):
    """
    class to deal with basic downloading/upload of objects to s3
    """
    def __init__(self):
        self.s3 = boto3.client('s3')

    def download_to_io(self, bucket, key, pandas=True):
        """
        function downloads a given file to a bytes object
        
        Args
        ----
        bucket: str
            string of the buvket you want to download from
        key: str
            string of the key/file name you want to download
        pandas: bool
            do you want it to return a pandas dataframe
        Returns
        -------
        df: dataframe
            the data from the file you're downloading
        fp: bytesio object
            the data from the file you're downloading
        """
        obj = self.s3.get_object(Bucket=bucket, Key=key)
        fp = BytesIO()
        fp.write(obj['Body'].read())
        fp.seek(0)
        if pandas:
            df = pd.read_csv(fp)
            return df
        else:
            return fp

    def download_to_file(self, bucket, key, savename):
        """
        function downloads a given file under savename
        
        Args
        ----
        bucket: str
            string of the buvket you want to download from
        key: str
            string of the key/file name you want to download
        savename: str
            the name of the file you want to save it to
        """
        self.s3.download_file(bucket, key, savename)
        print("file written to: '{}'".format(savename))
        #return True

    def sklearn_model_to_s3(self, model, bucket, key):
        """
        function upload an sklearn model to s3
        
        Args
        ----
        model: sklearn model
            used for the transformermixen
        bucket: str
            string of the buvket you want to upload to
        key: str
            string of the key/file name you want to upload to
        Returns
        -------
        bool if sucessful or not
        """
        fp = BytesIO()
        pickle.dump(model, fp, protocol=2)
        fp.seek(0)
        self.s3.upload_fileobj(fp, bucket, key)
        return True

    def sklearn_model_from_s3(self, bucket, key):
        """
        function upload an sklearn model to s3
        
        Args
        ----
        bucket: str
            string of the buvket you want to upload to
        key: str
            string of the key/file name you want to upload to
        savename: str
            the name of the file you want to save it to
        Returns
        -------
        model
        """
        obj = self.s3.get_object(Bucket=bucket, Key=key)
        model = cPickle.loads(obj['Body'].read())
        return model

    def upload_predictions_to_s3(self, df, bucket, key):
        """
        function upload the model's predictions to s3
        
        Args
        ----
        df: pandas dataframe
            the dataframe with the predictions you want to load
        bucket: str
            string of the buvket you want to upload to
        key: str
            string of the key/file name you want to upload to
        """
        #bucket = 'predictions'
        #key = "outputdata/output.csv"
        fp = BytesIO()
        df.to_csv(fp, index_label=False, index=False, header=True)
        # has to be some way to monitor progress
        self.s3.put_object(Bucket = bucket, Key = key, Body=fp.getvalue())

    def upload_file(self, fname, bucket, key):
        """
        basic function to upload a file to s3
        
        Args
        ----
        fname: str
            the file you want to upload to s3
        bucket: str
            string of the buvket you want to upload to
        key: str
            string of the key/file name you want to upload to
        """
        #bucket = 'predictions'
        #key = "outputdata/output.csv"
        self.s3.upload_file(fname, Bucket = bucket, Key = key)

    def upload_bytes(self, obj, bucket, key, use_json = True):
        """
        upload obj to an s3 bucket using the bytesio method
        instead of physically saving it to the device
        
        Args
        ----
        obj: the python variable/model/class/object
        bucket: str
            string of the buvket you want to upload to
        key: str
            string of the key/file name you want to upload to
        use_json: bool
            if true, use json to dump it, if not use pickle
        """
        fp = BytesIO()
        if use_json:
            json.dump(obj, fp)
        else:
            pickle.dump(obj, fp)
        fp.seek(0)
        self.s3.upload_fileobj(fp, bucket, key)

#---------------------------------------------------------------------
# model helpers
#---------------------------------------------------------------------

class TextTransformer(TransformerMixin):
    """
    Convert text to cleaned text using sklearn api
    for details on this class see: 
    http://scikit-learn.org/stable/developers/contributing.html#pipeline-compatibility
    """
    def __init__(self):
        # List of symbols we don't care about
        self.SYMBOLS = " ".join(punctuation).split(" ") + ["-----", "---", "...", "“", "”", "'ve"]

        # tools for splitting text
        # TODO(krzum) add support for this in py3
        self.punctuation = punctuation.replace("'",'').replace('"','')
        self.trans = maketrans(punctuation, ' '*len(punctuation))
    
    def transform(self, X, **transform_params):
        return [self.clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}
    
    def clean_text(self, text):
        """
        custom text that removes waste from text
        Args
        ----
        text: str
            the text you want cleaned
        Returns
        -------
        list of word tokens 
        """
        # get rid of newlines and 'unknown' and 'NaN'
        text = text.strip().replace("\n", " ").replace("\r", " ")
        # replace HTML symbols
        text = text.replace("&amp;", "and").replace("&gt;", ">").replace("&lt;", "<")
        # lowercase
        text = text.lower()
        # remove nonascii characters
        text = self.remove_non_ascii(text)
        tokens = text.encode('ascii').lower().translate(self.trans).split()
        tokens = [tok for tok in tokens if tok not in self.SYMBOLS]
        # reassemble text
        return " ".join(tokens)

    def remove_non_ascii(self,s):
        """
        remove non ascii characters from unicode text
        Args:
        -----
        s: str
            the string you want to remove nonascii characters from
        Returns:
        -------
            string with non-ascii characters removed

        """
        return "".join(filter(lambda x: ord(x)<128, s))


    def create_count_index(self, data, clean = True):
        """
        Creates a dictionary of the words in the dataset and 
        the count of each word
    
        Args:
        -----
        data: array
            iterable of strings
        clean: bool
            to perform cleaning on the text
        Returns:
        --------
        word_dict: dict
            {word: count}
        """
        word_counter = Counter()
        for line in data:
            if clean:
                word_counter.update(self.clean_text(line).split())
            else:
                word_counter.update(line.split())
        idx = 1
        self.word_dict = {}
        for word in word_counter.most_common():
            self.word_dict[word[0]] = idx
            idx+=1
        self.idx2word = {v: k for k, v in self.word_dict.items()}
        self.idx2word[0] = ' '

    def encode_dataset(self, data, max_vocab = 20000, max_seq_len = 500, 
                       value = 0):
        """
        function that encodes a dataset into sequence of numbers. 
        This funciton will fail unless create_count_index was run 
        prior to this

        Args:
        -----
        data: array, list of strings
            the data you want encoded to numbers
        max_vocab: int
            the max number of most frequent vocab words to include
        max_seq_len: int
            the maximum sequence length
        value: int
            the value to give to the sentences that do not meet the
            max_seq_len
        Returns:
        --------
        output: np.array
            array of shape len(data) by max_seq_le
        """
        self.max_vocab = max_vocab
        assert(type(self.word_dict) == dict)
        output = np.zeros((len(data),max_seq_len))
        for i in range(len(data)):
            line = data[i]
            cleaned_sentences = self.clean_text(line).split()
            # encode the words in the dataset
            output_line = []
            for word in cleaned_sentences:
                if word in self.word_dict:
                    output_line.append(self.word_dict[word])
            output_line = np.array(output_line)

            output_line[output_line > max_vocab-1] = (max_vocab - 1)

            if(len(output_line) > max_seq_len):
                new_sentence = output_line[:max_seq_len]
                output[i] = new_sentence
            else:
                num_padding = max_seq_len - len(output_line)
                new_sentence = np.append(output_line,[value] * num_padding)
                output[i] = new_sentence
        return output

    def decode_sentences(self, data):
        """
        function to decode a sequence of numbers to strings

        Args:
        -----
        data : array of ints/floats
            the sequence of data you want decoded
        Returns:
        --------
        output_strings: list(strings)
            the decoded dataset
        """
        output_strings = []
        for line in data:
            output_line = [self.idx2word[int(idx)] for idx in line]
            output_strings.append(" ".join(output_line).strip())
        return output_strings

    def create_spacy_embedding(self, vocab_size = None):
        """
        function to create an embedding matrix from spacy's
        glove vectors

        Args: 
        -----
        vocab_size: int or None
            if the size is int, it will create an embedding 
            matrxi of vocab_size by 300 (if spacy en-core-web-md 
            is used)
            to avoid errors, the model defaults to self.max_vocab
            size so that when you use with the ebedding initialization
            it doesn't throw errors
        Returns:
        --------
        embedding_matrix: mx.ndarray
            the ebedding matrix of size (self.max_vocab or self.max_vocab) 
            by spacy's vector shape
            This matrix will be used to initialize the embedding matrix
            of the LSTM
        """
        import spacy
        if not vocab_size:
            vocab_size = self.max_vocab
        # have to make sure this is where it's located
        # print('loading spacy data...')
        try:
            nlp = spacy.load('en_core_web_md')
        except:
            print('error - need to download en_core_web_md')
            # this was happening on the deeplearning ami
            os.system("python2 -m spacy download en_core_web_md")
            nlp = spacy.load('en_core_web_md')
        apple_id = nlp.vocab.strings['apple']
        num_embed = nlp.vocab[apple_id].vector.shape[0]
        embedding_matrix = np.zeros((vocab_size, num_embed))
        for word, i in self.word_dict.items():
            if i >= vocab_size:
                continue
            word_id = nlp.vocab.strings[word]
            try:
                if nlp.vocab[word_id].has_vector:
                    embedding_vector = nlp.vocab[word_id].vector
                    embedding_matrix[i] = embedding_vector
                else:
                    pass
            except:
                pass
        embedding_matrix = mx.nd.array(embedding_matrix)
        return embedding_matrix

#---------------------------------------------------------------------
# mxnet models
#---------------------------------------------------------------------

class LSTM(gluon.Block):
    """A model with an encoder, recurrent layer, and a decoder."""
    def __init__(self, n_classes=2, input_size=384, 
                      embed_size=300, num_hidden=100, 
                      num_layers=3, dropout=0.5, 
                      bidirectional=True, **kwargs):

        super(LSTM, self).__init__(**kwargs)
        with self.name_scope():
            self.drop = nn.Dropout(dropout)
            self.encoder = nn.Embedding(input_size, embed_size, 
                        weight_initializer = mx.init.Uniform(0.1))
            self.lstm = rnn.LSTM(num_hidden, num_layers=num_layers, 
                            bidirectional=bidirectional, dropout=dropout, 
                            input_size=embed_size, layout = 'NTC')
            self.decoder = nn.Dense(n_classes)
            self.num_hidden = num_hidden
            self.n_classes = n_classes

    def forward(self, inputs, hidden):
        inputs = self.encoder(inputs)
        emb = self.drop(inputs)
        output, hidden = self.lstm(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden

    def begin_state(self, *args, **kwargs):
        return self.lstm.begin_state(*args, **kwargs)


def detach(hidden):
    # helper function for LSTM
    if isinstance(hidden, (tuple, list)):
        hidden = [i.detach() for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden   

class CNN(gluon.Block):
    """A model with a CNN, embedding, and Pool layer"""
    def __init__(self, n_classes   = 2, 
                       kernel_size = 8, 
                       embed_size  = 300, 
                       dropout     = 0.5, 
                       seq_len     = 500, 
                       vocab_size  = 7500, 
                       **kwargs):
        super(CNN, self).__init__(**kwargs)
        self.dropout = False
        with self.name_scope():
            self.encoder = nn.Embedding(vocab_size, embed_size, 
                        weight_initializer = mx.init.Uniform(0.1))
            self.conv = nn.Conv2D(embed_size, (kernel_size, embed_size),1)
            self.act = nn.Activation('relu')
            self.pool = nn.MaxPool2D((seq_len-kernel_size+1, 1))
            if dropout > 0.0:
                self.dropout = True
                self.drop = nn.Dropout(dropout)
            self.decoder = nn.Dense(n_classes)

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = x.expand_dims(1)
        x = self.conv(x)
        x = self.act(x)
        x = self.pool(x)
        if self.dropout:
            x = self.drop(x)
        x = self.decoder(x)
        return x

class DataGen(gluon.data.Dataset):
    # standard data generator for gluon
    def __init__(self, data, labels):
        """
        :param inputs: a list of numpy array
        :param labels: a list of numpy array
        """
        self.data = data
        self.labels = labels
        assert(data.shape[0]==labels.shape[0])

    def __getitem__(self, item):
        x = self.data[item]
        y = self.labels[item]
        return x, y 

    def __len__(self):
        return len(self.data)

#---------------------------------------------------------------------
# evaluation helpers
#---------------------------------------------------------------------

def evaluate_cnn(data_iterator, loss, net, ctx):
    # evaluator for cnn
    total_L = 0.0
    ntotal = 0
    acc = mx.metric.Accuracy()
    # get batchsize
    for x, y in data_iterator:
        break
    batch_size = x.shape[0]
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1, keepdims=True) #RYAN
        acc.update(preds=predictions, labels=label)
        L = loss(output, label)
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
    return acc.get()[1], total_L / ntotal

def evaluate(data_iterator, loss, net, ctx):
    # evaluator for an lstm 
    total_L = 0.0
    ntotal = 0
    acc = mx.metric.Accuracy()
    # get batchsize
    for x, y in data_iterator:
        break
    batch_size = x.shape[0]
    hidden = net.begin_state(func = mx.nd.zeros, batch_size = batch_size, ctx=ctx)
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output, hidden = net(data, hidden)
        predictions = nd.argmax(output, axis=1, keepdims=True) #RYAN
        acc.update(preds=predictions, labels=label)
        L = loss(output, label)
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
    return acc.get()[1], total_L / ntotal

def perplexity(L):
    try:
        perplexity = math.exp(L)
        if perplexity > 100:
            perplexity = float('inf')                
    except OverflowError:
        perplexity = float('inf')
    return perplexity


