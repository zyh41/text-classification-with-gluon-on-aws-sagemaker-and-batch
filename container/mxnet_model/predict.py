#!/usr/local/bin/python2
# -*- coding: utf-8 -*-
import pandas as pd
import sklearn
import numpy as np
import boto3
from io import BytesIO
from io import StringIO
import os
import sys
import argparse
import mxnet as mx
import json
from utils import *
import subprocess   

parser = argparse.ArgumentParser()
parser.add_argument('--DATAURL', type = str, 
                    help = 'the name of the bucket that contains the data')
parser.add_argument('--KEY', type = str, 
                    help = 'expected in a .csv format')
parser.add_argument('--MODELCONFIGURL', type = str, 
                    help = 'the json config for training your model')
parser.add_argument('--TRANSFORMERURL', type = str, 
                    help = 'the sklearn pipeline for performing text transforms')
parser.add_argument('--MODELPARAMSURL', type = str, 
                    help = 'the model.params file that contains the trained weights')
parser.add_argument('--S3BUCKETRESULTS', type = str, 
                    help = 'the bucket location where you wish to dump your predictions to')
args = parser.parse_args()

if __name__ == '__main__':
    dm = DownloadManager()
    context = mx.gpu()

    print('[INFO] Downloading data')
    data = dm.download_to_io(bucket = args.DATAURL, 
                             key    = args.KEY, 
                             pandas = True)
    
    print('[INFO] Downloading config data')
    fp = dm.download_to_io(bucket = args.MODELCONFIGURL,
                           key    = 'config.json', 
                           pandas = False)
    config = json.loads(fp.read())
    convert_config(config)

    print('[INFO] Downloading weights')
    #'/tmp/model.params', 
    dm.download_to_file(savename = '/tmp/'+config['save_name'], 
                        bucket   = args.MODELPARAMSURL,
                        key      = config['save_name'])
    
    if config["cnn"]:
        print('[INFO] Building CNN')
        net = CNN(n_classes   = config['n_classes'],
                  kernel_size = config['kernel_size'], 
                  embed_size  = config['embed_size'], 
                  dropout     = config['dropout'], 
                  seq_len     = config['seq_len'],
                  vocab_size  = config['vocab_size'])
    else:
        print('[INFO] Building LSTM')
        net = LSTM(n_classes     = config['n_classes'],
                   input_size    = int(config['vocab_size']), 
                   embed_size    = config['embed_size'],
                   num_hidden    = int(config['hidden_size']), 
                   num_layers    = int(config['n_layers']), 
                   dropout       = float(config['dropout']), 
                   bidirectional = config['bidirectional'])

    net.load_params('/tmp/'+config['save_name'], context)

    if not config["cnn"]:
        hidden = net.begin_state(func       = mx.nd.zeros, 
                                 batch_size = data_encoded.shape[0], 
                                 ctx        = context)

    print('[INFO] Downloading transformer')
    transformer = dm.sklearn_model_from_s3(args.TRANSFORMERURL, key='transformer.pkl')
    
    print('[INFO] Transforming data and loading data  generator')
    cleaned_data = transformer.transform(data.review.values)
    data_encoded = transformer.encode_dataset(data        = cleaned_data, 
                                              max_vocab   = config['vocab_size'], 
                                              max_seq_len = config['seq_len'])
    data_encoded = mx.nd.array(data_encoded, ctx = context)
    data_generator =  mx.gluon.data.DataLoader(data_encoded, 
                                               batch_size = config['batch_size'], 
                                               shuffle    = False)
    print('[INFO] Making predictions')
    predictions = []
    for data_batch in data_generator:
        if not config["cnn"]:
            hidden = detach(hidden)
            preds, hidden = net(data_batch, hidden)
        else:
            preds = net(data_batch)
        predictions.extend(list(preds.asnumpy().argmax(axis=1)))

    assert(len(predictions) == data.shape[0])
    print('[INFO] Predictions successful')
    data['predictions'] = predictions

    print('[INFO] Uploading predictions to s3')
    name = args.KEY.replace('.csv','')
    dm.upload_predictions_to_s3(df    = data, 
                               bucket = args.S3BUCKETRESULTS,
                                key   = name+'_output.csv')
    print('[INFO] Predictions loaded to {}!'.format(args.S3BUCKETRESULTS))


