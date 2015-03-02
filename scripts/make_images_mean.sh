#!/usr/bin/env sh
# Compute the mean image from the images 

../../caffe/build/tools/compute_image_mean data/64_bal/train_lmdb data/64_bal/train_mean.binaryproto

../../caffe/build/tools/compute_image_mean data/64_bal/val_lmdb data/64_bal/val_mean.binaryproto

../../caffe/build/tools/compute_image_mean data/64_bal/test_lmdb data/64_ba;/test_mean.binaryproto

echo "Done."
