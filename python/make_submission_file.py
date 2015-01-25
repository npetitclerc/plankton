"""
Create the prediction file to submit on Kaggle

"""

import numpy as np
import caffe
import os

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = 'caffe/deploy_alexnet.prototxt'
PRETRAINED = 'caffe/256_padded/alexnet_snapshots_iter_20000.caffemodel'
IMAGES_FOLDER = 'data/256_padded/test/'
MEAN_FILE = 'data/256_padded/test_mean.npy' # Converted with convert_protomean.py
INDEX_FILE = "data/plankton_index.csv"
SUBMISSION_FILE = "caffe/256_padded/submission.csv"

caffe.set_phase_test()
caffe.set_mode_cpu()

net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(MEAN_FILE),
                       image_dims=(256, 256),
                       )
                       
indexes = pd.read_csv(INDEX_FILE, header=None)
submission = open(SUBMISSION_FILE, 'w')
header = ['image'] + indexes.values[:,1]
print >> submission, ",".join(header)

images = os.listdir(IMAGES_FOLDER)
predictions = net.predict(images[:10])  # predict takes any number of images, and formats them for the Caffe net automatically
for i, pred in enumerate(predictions):
  print 'prediction shape:', pred.shape
  print 'predicted class:', pred.argmax()
  print 'prediction class:', indexes.iloc[pred.argmax()].values[1]
  print >> submission, ",".join([images[i]] + list(pred))
  
submission.close()

 
