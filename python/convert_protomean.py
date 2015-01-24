""" From: https://github.com/BVLC/caffe/issues/290
Convert proto mean files to a python array.
The Python script to predict new images wants a python array for the means, but the means are saved as proto file...

Usage: 
python convert_protomean.py proto.mean out.npy 

"""

import caffe
import numpy as np
import sys

if len(sys.argv) != 3:
    print "Usage: python convert_protomean.py proto.mean out.npy"
    sys.exit()

blob = caffe.proto.caffe_pb2.BlobProto()
data = open( sys.argv[1] , 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
out = arr[0]
np.save( sys.argv[2] , out )

