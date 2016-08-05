# -*- coding: utf-8 -*-
import numpy as np
from skimage import io
import os


def convert(fname):
  image = io.imread(fname, as_grey=True)
  unique_dict = {k:i for i, k in enumerate(np.unique(image))}
  outs = [np.zeros(image.shape, dtype=np.uint8) for _ in range(len(unique_dict))]
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      channel_index = unique_dict[image[i,j]]
      outs[channel_index][i,j] = 255

  dname = os.path.dirname(fname)
  for i, out in enumerate(outs):
    outname = os.path.join(dname, "class%d.tif" % i)
    print("Saving file %s" % outname)
    io.imsave(outname, out, plugin='tifffile')

if __name__ == "__main__":
  parent = "/Users/joshuaarnold/Documents/Papers/VU_SEM/analysis/" \
           "SEM-EDX-DATA/BG2/"
  print(os.listdir(parent))

  for i in range(1,13):
    fname = os.path.join(parent,
                         "soi_%s/Classified image.tif" % str(i).zfill(3)
                         )
    convert(fname)