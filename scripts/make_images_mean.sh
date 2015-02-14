#!/usr/bin/env sh
# Compute the mean image from the images 

../../caffe/build/tools/compute_image_mean data/64_sp/train_lmdb data/64_sp/train_mean.binaryproto

../../caffe/build/tools/compute_image_mean data/64_sp/val_lmdb data/64_sp/val_mean.binaryproto

../../caffe/build/tools/compute_image_mean data/64_sp/test_lmdb data/64_sp/test_mean.binaryproto

echo "Done."
