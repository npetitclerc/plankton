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
MODEL_FILE = 'caffe/64_aug/stride1/deploy_alexnet.prototxt'
PRETRAINED = 'caffe/64_aug/stride1/snapshots_iter_12000.caffemodel'
IMAGES_FOLDER = 'data/64_aug/val/'
MEAN_FILE = 'data/64_aug/val_mean.npy' # Converted with convert_protomean.py
INDEX_FILE = "data/plankton_index.csv"
Y_FILE = 'data/64_aug/val.txt'
VAL_FILE = "caffe/64_aug/stride1/val_30k.csv"
MEAN_ERR_FILE = 'caffe/64_aug/stride1/mean_err.pkl'
MEAN_NERR_FILE = 'caffe/64_aug/stride1/mean_nerr.pkl'  

batch_size = 2500 # Process images by batch if memory is an issue 

caffe.set_phase_test()
caffe.set_mode_gpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(MEAN_FILE),
                       image_dims=(64, 64),
                       raw_scale=255,
                       gpu=True
                       )
                       
indexes = pd.read_csv(INDEX_FILE, header=None).values
y_val = pd.read_csv(Y_FILE, header=None, sep=' ').values
val = open(VAL_FILE, 'w')
header = ['image'] + list(indexes.values[:,1])
print >> val, ",".join(header)

images_f = os.listdir(IMAGES_FOLDER)
n_im = len(images_f)

n_classes = len(indexes)
val_per_class = np.zeros(n_classes) # Number of validation images per class
err_sum = np.zeros(n_classes) # accumulate the error per class
num_err = np.zeros(n_classes) # accumulate the number of errors per class

for ibatch in xrange(0, n_im, batch_size):
  print "Processing image ", ibatch, "/", n_im, " => ", "%.2f%%" % (100.0 * ibatch / n_im)
  images =[caffe.io.load_image(IMAGES_FOLDER + im, color=False) for im in images_f[ibatch:ibatch + batch_size]]
  predictions = net.predict(images)  # predict takes any number of images, and formats them for the Caffe net automatically

  for ipred, pred in enumerate(predictions):
    sol = np.zeros(n_classes)
    sol[y_val[ipred, 1]] = 1.
    err_sum += abs(pred - sol)
    val_per_class[y_val[ipred, 1]] += 1
    if pred.argmax() != y_val[ipred, 1]:
      num_err[pred.argmax()] += 1
      num_err[y_val[ipred, 1]] += 1
    print >> val, ",".join([images_f[ibatch + ipred]] + pred)

val.close()
  
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
  
  
