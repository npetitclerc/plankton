#!/usr/bin/env sh
# Compute the mean image from the images 

../../caffe/build/tools/compute_image_mean data/64_aug/train_lmdb data/64_aug/train_mean.binaryproto

../../caffe/build/tools/compute_image_mean data/64_aug/val_lmdb data/64_aug/val_mean.binaryproto

../../caffe/build/tools/compute_image_mean data/64_aug/test_lmdb data/64_aug/test_mean.binaryproto

echo "Done."
