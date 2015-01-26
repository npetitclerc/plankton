"""
Create the prediction file to submit on Kaggle

"""

import numpy as np
import caffe
import os
import pandas as pd

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = 'caffe/deploy_alexnet.prototxt'
#MODEL_FILE = 'caffe/deploy_alexnet_old.prototxt'
PRETRAINED = 'caffe/256_scalepad/alexnet_snapshots_scalepad_iter_20000.caffemodel'
#PRETRAINED = 'caffe/256_padded/alexnet_snapshots_iter_20000.caffemodel'
IMAGES_FOLDER = 'data/256_scalepad/test/'
MEAN_FILE = 'data/256_scalepad/test_mean.npy' # Converted with convert_protomean.py
INDEX_FILE = "data/plankton_index.csv"
SUBMISSION_FILE = "caffe/256_scalepad/submission.csv"
batch_size = 2000 # Process images by batch if memory is an issue 

caffe.set_phase_test()
caffe.set_mode_gpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(MEAN_FILE),
                       image_dims=(256, 256),
                       raw_scale=255,
                       gpu=True
                       )
                       
indexes = pd.read_csv(INDEX_FILE, header=None)
submission = open(SUBMISSION_FILE, 'w')
header = ['image'] + list(indexes.values[:,1])
print >> submission, ",".join(header)

images_f = os.listdir(IMAGES_FOLDER)
n_im = len(images_f)

for ibatch in xrange(0, n_im, batch_size):
  print "Processing image ", ibatch, "/", n_im, " => ", "%.2f%%" % (100.0 * ibatch / n_im)
  images =[caffe.io.load_image(IMAGES_FOLDER + im, color=False) for im in images_f[ibatch:ibatch + batch_size]]

  predictions = net.predict(images)  # predict takes any number of images, and formats them for the Caffe net automatically
  pp = np.zeros(batch_size) 
  for ipred, pred in enumerate(predictions):
    pp[ipred] = pred.argmax()    
    pred = [str(p) for p in pred]
    print >> submission, ",".join([images_f[ibatch + ipred]] + pred)
  print 'predicted classes:', pp
submission.close()
print "Done. See results in ", SUBMISSION_FILE
 
