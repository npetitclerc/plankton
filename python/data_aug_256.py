""" Augment the number of images by keeping the class imbalance - making random affine transformations
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
  nimg = ''.join([os.path.splitext(img)[0], append, os.path.splitext(img)[1]]) 
  new_file = "/".join([out_folder, nimg])
  try:
    shutil.copy("/".join([in_folder, d, img]), new_file)
    print >> print_file, " ".join([nimg, planktons[d]])

    rotation = random.randrange(1, 360)
    scale = random.uniform(0.7, 1.3)
    shift_x = random.randrange(-20, 20) + 128
    shift_y = random.randrange(-20, 20) + 128   
    cmd = ['convert',
        '-resize', '256x256!',
        #'-resize', '256x256',
        #'-gravity', 'center',
        #'-extent', '256x256',
        '-distort', 'SRT', '128,128,%.4s,%s,%s,%s'%(scale, rotation, shift_x, shift_y),
        new_file, 
        new_file]
    p = subprocess.Popen(cmd)
  except:
    print "Crashed!"
    print "Working on: ", new_file
    p = None
  return p

def scale_test_images(img, in_folder, out_folder, d, print_file, planktons):
  """ Create a process to make convert a test image """  
  new_file = "/".join([out_folder, img])
  try:
    shutil.copy("/".join([in_folder, d, img]), new_file)
    print >> print_file, " ".join([img, planktons[d]])
 
    cmd = ['convert',
        '-resize', '256x256!',
        #'-resize', '256x256',
        #'-gravity', 'center',
        #'-extent', '256x256',
        new_file, 
        new_file]
    p = subprocess.Popen(cmd)
  except:
    print "Crashed! - test image"
    print "Working on: ", new_file
    p = None
  return p

# Paths
submission_file = "data/raw/sampleSubmission.csv"
index_file_path = "data/plankton_index.csv"
output_folder = "data/256_scale_aug2"
#input_folder = output_folder + "/train_all"
input_folder = "data/raw/train"
train_list = output_folder + "/train.txt"
val_list = output_folder + "/val.txt"
test_list = output_folder + "/test.txt"
train_folder = output_folder + "/train"
val_folder = output_folder + "/val" 
test_folder = output_folder + "/test"

#nimg_per_class = 2000
aug_factor = 5
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
  # Splitting 20% of each class for validation 
  nimgs = len(imgs)
  random.shuffle(imgs)
  split = int(split_ratio * nimgs)
  imgs_train = imgs[:split]
  imgs_val = imgs[split:]
   
  n_im = 0
  append = ''
  n_im_max = split_ratio * nimgs * aug_factor
  while n_im < n_im_max: 
    processes = []
    for img in imgs_train:
      p = apply_random_transform(img, input_folder, train_folder, append, d, train_file, planktons)
      processes.append(p)
      n_im += 1
      if n_im == n_im_max:
        break
    append += '_'  
    for p in processes:
      p.wait()
      
  processes = []
  for img in imgs_val:
    p = scale_test_images(img, input_folder, val_folder, d, val_file, planktons)
    processes.append(p)
    for p in processes:
      p.wait()

train_file.close()
val_file.close()  

# List the test images
# ln -s data/64_sp/test data/64_bal/test
test_file = open(test_list, "w")
for img in os.listdir(test_folder):
  print >> test_file, " ".join([img, "0"])

test_file.close()

