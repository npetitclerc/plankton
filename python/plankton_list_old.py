import pandas as pd
import os
import shutil

# List the plankton and index values
sample = pd.read_csv("sampleSubmission.csv")
sample = pd.read_csv("../plankton/sampleSubmission.csv")
classes = sample.columns.values[1:]
list_file = open("plankton_list.csv", 'w')
planktons = {}
for i, c in enumerate(classes):
  print >> list_file, ",".join([str(i) , c])
  planktons[c] = str(i)
list_file.close()


# List the images and index and copy files to a common directory
folder = "train_256_padded"
all_imgs = folder + "_all"
if not os.path.exists(all_imgs):
  os.mkdir(all_imgs)
file_images = open(folder + ".txt", "w")

for d in os.listdir(folder):
  for img in os.listdir("/".join([folder, d])):
    #shutil.copyfile("/".join([folder, d, img]), "/".join([all_imgs, img]))
    print >> file_images, " ".join([img, planktons[d]])
file_images.close()
  
