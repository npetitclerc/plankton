""" Split the train set in train and val sets.
    Create the list of images with labels. 
    Should be run from the root (plankton) folder.
"""

import pandas as pd
import os
import shutil
import random

# Paths
submission_file = "data/raw/sampleSubmission.csv"
index_file_path = "data/plankton_index.csv"
output_folder = "data/256_padded"
input_folder = output_folder + "/train_all"
train_list = output_folder + "/train.txt"
val_list = output_folder + "/val.txt"
test_list = output_folder + "/test.txt"
train_folder = output_folder + "/train"
val_folder = output_folder + "/val" 
test_folder = output_folder + "/test"

split_ratio = 0.8 # Fraction of train set kept for training - the rest goes to validation

# List the plankton and index values
sample = pd.read_csv(submission_file)
classes = sample.columns.values[1:]
index_file = open(index_file_path, 'w')
planktons = {}
for i, c in enumerate(classes):
  print >> index_file, ",".join([str(i) , c])
  planktons[c] = str(i)
index_file.close()


# List the images and index and copy files to a common directory
if not os.path.exists(output_folder):
  os.mkdir(output_folder)
if not os.path.exists(train_folder):
  os.mkdir(train_folder)
if not os.path.exists(val_folder):
  os.mkdir(val_folder)

train_file = open(train_list, "w")
val_file = open(val_list, "w")

for d in os.listdir(input_folder):
  imgs = os.listdir("/".join([input_folder, d]))
  random.shuffle(imgs)
  split = int(split_ratio * len(imgs))
  imgs_train = imgs[:split]
  imgs_val = imgs[split:]
  
  for img in imgs_train:
    shutil.move("/".join([input_folder, d, img]), "/".join([train_folder, img]))
    print >> train_file, " ".join([img, planktons[d]])
  for img in imgs_val:
    shutil.move("/".join([input_folder, d, img]), "/".join([val_folder, img]))
    print >> val_file, " ".join([img, planktons[d]])

train_file.close()
val_file.close()  

# List the test images
test_file = open(test_list, "w")
for img in os.listdir(test_folder):
  print >> test_file, " ".join([img, "0"])

test_file.close()

