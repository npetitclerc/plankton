"""
Look at the validation results, to assess the classification

"""

import numpy as np
import caffe
import os, time
import pandas as pd
import pickle

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
VAL_FILE = "caffe/64_aug/stride1/val_30k.csv"
INDEX_FILE = "data/plankton_index.csv"
Y_FILE = 'data/64_aug/val.txt'

MEAN_ERR_FILE = 'caffe/64_aug/stride1/mean_err.pkl'
MEAN_NERR_FILE = 'caffe/64_aug/stride1/mean_nerr.pkl'  

indexes = pd.read_csv(INDEX_FILE, header=None).values
y_val = pd.read_csv(Y_FILE, header=None, sep=' ').values
val_pred = pd.read_csv(VAL_FILE, skiprows=1).values

n_im = len(y_val)
n_classes = len(indexes)

val_per_class = np.zeros(n_classes) # Number of validation images per class
err_sum = np.zeros(n_classes) # accumulate the error per class
num_err = np.zeros(n_classes) # accumulate the number of errors per class

for ipred, pred in enumerate(val_pred):
  sol = np.zeros(n_classes)
  sol[y_val[ipred, 1]] = 1.
  err_sum += abs(pred - sol)
  val_per_class[y_val[ipred, 1]] += 1
  if pred.argmax() != y_val[ipred, 1]:
    num_err[pred.argmax()] += 1
    num_err[y_val[ipred, 1]] += 1
  
mean_err = err_sum / val_per_class
mean_nerr = num_err / val_per_class * 100.

print "Mean Error:"
m_sort = np.argsort(mean_err)[::-1]
for i in m_sort:
  print indexes[i, 1], mean_err[i]

print "Mean Number of Error (%):"
m_sort = np.argsort(mean_nerr)[::-1]
for i in m_sort:
  print indexes[i, 1], mean_nerr[i] 
  
f = open(MEAN_ERR_FILE, 'wb')
pickle.dump(mean_err, f)
f.close()
f = open(MEAN_NERR_FILE, 'wb')
pickle.dump(mean_nerr, f)
f.close()
  
  
