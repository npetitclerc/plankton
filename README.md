# plankton
Note: Every scripts and python codes should be run from the project root (aka. plankton/) directory.

Order of steps:
## Preprocessing
1 - convert_images.sh - Convert images to desired size, but still in .jpg

2 - plankton_list.py - To create the list of files and labels, it also split the train, val and test sets

3 - create_image_db.sh - Convert the images to lmdb format

4 - make_images_mean.sh - Finds the images mean

5 - convert_protomean.py - Convert test protomean to .npy format

python python/convert_protomean.py data/256_scale/test_mean.binaryproto data/256_scale/test_mean.npy

6 - Then can delete /data/xxx/test_lmdb

## Training
5 - Train:
Edit:

caffe/solver_alexnet.prototxt

Run:

../../caffe/build/tools/caffe train --solver=caffe/solver_alexnet.prototxt 2>&1 | tee caffe/...

../../caffe/build/tools/caffe train --solver=caffe/64_aug/stride2/solver_alexnet.prototxt 2>&1 | tee caffe/64_aug/stride2/train.log

if restarting from a snapshot use: --snapshot=...

##Monitor: 

watch nvidia-smi

6 - Test:

python python/make_submission_file.py


Test Caffe speed:

build/tools/caffe time --model=models/bvlc_alexnet/deploy.prototxt --gpu=0
