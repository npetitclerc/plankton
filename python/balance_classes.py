""" Balance the number of images by class by data augmentation - making random affine transformations
    Split the train set in train and val sets.
    Create the list of images with labels. 
    Should be run from the root (plankton) folder.
"""

import pandas as pd
import os, subprocess
import shutil
import random

def apply_random_transform(img, in_folder, out_folder, append, d, print_file, planktons):
  """ Create a process to make new image with a random affine transformation """  
  nimg = ''.join([os.split(img)[0], append, os.split(img)[1]]) 
  new_file = "/".join([out_folder, nimg])
  shutil.copy("/".join([in_folder, d, img]), new_file)
  
  rotation = random.randrange(1, 360)
  scale = random.uniform(0.7, 1.3)
  shift_x = random.randrange(-5, 5)
  shift_y = random.randrange(-5, 5)
  
  cmd = ['convert',
        '-resize', '64x64',
        '-gravity', 'center',
        '-extent', '64x64',
        '-distort', 'SRT', '32,32,%.4s,%s,%s,%s'%(scale, rotation, shift_x, shift_y),
        new_file, 
        new_file]

  #convert -resize 64x64 -gravity center -affine .9,-.1,.1,.9,0,0 -transform -extent 64x64 data/raw/train/tunicate_doliolid/1107.jpg toto2.jpg
  p = subprocess.Popen(cmd)
  print >> print_file, " ".join([nimg, planktons[d]])
  return p

# Paths
submission_file = "data/raw/sampleSubmission.csv"
index_file_path = "data/plankton_index.csv"
#output_folder = "data/256_scalepad"
#output_folder = "data/256_scale"
output_folder = "data/64_bal"
input_folder = output_folder + "/train_all"
train_list = output_folder + "/train.txt"
val_list = output_folder + "/val.txt"
test_list = output_folder + "/test.txt"
train_folder = output_folder + "/train"
val_folder = output_folder + "/val" 
test_folder = output_folder + "/test"

nimg_per_class = 2000
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
   
  n_im = 0
  append = ''
  while n_im < split_ratio * nimg_per_class: 
    processes = []
    for img in imgs_train:
      p = apply_random_transform(img, input_folder, train_folder, append, d, train_file, planktons)
#      nimg = ''.join([os.split(img)[0], append, os.split(img)[1]]) 
#      new_file = "/".join([train_folder, nimg])
#      shutil.copy("/".join([input_folder, d, img]), new_file)
#      # Apply random transformation
#      cmd = ['convert',
#            '-resize', '64x64',
#            '-gravity', 'center',
#            '-affine', '.9,-.1,.1,.9,0,0', '-transform',
#            '-extent', '64x64',
#            new_file, 
#            new_file]
#      #convert -resize 64x64 -gravity center -affine .9,-.1,.1,.9,0,0 -transform -extent 64x64 data/raw/train/tunicate_doliolid/1107.jpg toto2.jpg
#      p = subprocess.Popen(cmd)
      processes.append(p)
            
      #print >> train_file, " ".join([nimg, planktons[d]])
      n_im += 1
      if n_im == split_ratio * nimg_per_class:
        break
    append += '_'  
    for p in processes:
      p.wait()
      
  n_im = 0
  append = ''    
  while n_im < (1. - split_ratio) * nimg_per_class: 
    processes = []
    for img in imgs_val:
      p = apply_random_transform(img, input_folder, val_folder, append, d, val_file, planktons)
      processes.append(p)
      n_im += 1
      if n_im == split_ratio * nimg_per_class:
        break
    append += '_'  
    for p in processes:
      p.wait()

train_file.close()
val_file.close()  

# List the test images
test_file = open(test_list, "w")
for img in os.listdir(test_folder):
  print >> test_file, " ".join([img, "0"])

test_file.close()

