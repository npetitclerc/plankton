#!/usr/bin/env sh
# Compute the mean image from the images 

../../caffe/build/tools/compute_image_mean data/256_scalepad/train_lmdb data/256_scalepad/train_mean.binaryproto

../../caffe/build/tools/compute_image_mean data/256_scalepad/val_lmdb data/256_scalepad/val_mean.binaryproto

../../caffe/build/tools/compute_image_mean data/256_scalepad/test_lmdb data/256_scalepad/test_mean.binaryproto

echo "Done."
