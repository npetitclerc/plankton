#!/usr/bin/env sh
# Compute the mean image from the images 

../../caffe/build/tools/compute_image_mean data/256_padded/train_lmdb data/256_padded/train_mean.binaryproto

../../caffe/build/tools/compute_image_mean data/256_padded/val_lmdb data/256_padded/val_mean.binaryproto

../../caffe/build/tools/compute_image_mean data/256_padded/test_lmdb data/256_padded/test_mean.binaryproto

echo "Done."
