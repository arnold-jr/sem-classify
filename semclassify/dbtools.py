# -*- coding: utf-8 -*-
from semclassify.helpers import stopwatch, get_input_json, make_input_json
import numpy as np
import pandas as pd
from skimage import io
import mahotas as mh
import os
from itertools import chain

class Material():
  def __init__(self, name, path, sites):
    """ Creates a new instance of Material

    :param name: unique identifier of the Material, corresponding to top-level
    directory name
    :param sites: list of sites
    """
    self.name = name
    self.path = path
    self.sites = sites

  def get_mask_names(self):
    """ Collects all mask names for each site.

    return: unique set of mask names
    """
    return list(set(chain.from_iterable(site.get_mask_names()
                                        for site in self.sites)) - {None})

  def get_dataframe(self):
    """ Returns a DataFrame for each Material
    :param None
    :return df: pandas DataFrame associated with the Material."""
    with stopwatch("creating DataFrame for material %s" % self.name):
      df = pd.DataFrame()
      acc = 0
      for s in self.sites:
        chunk = s.get_image_block()
        chunk.index += acc
        acc += len(chunk.index)
        df = df.append(chunk)
      df['material'] = self.name

      # if a mask is not present, then assume false
      df[self.get_mask_names()] = df[self.get_mask_names()].fillna(False)
      return df

  def __str__(self):
    return self.name + "->\n\t" + "\n\t".join(str(x) for x in self.sites)


class Site():
  def __init__(self, name, path, images):
    """ Creates a new site, aka field of view, where images have been collected

    :param name: unique identifier of this Site
    :param path: filepath
    :param images: list of Images acquired at this Site
    """
    self.name = name
    self.path = path
    self.images = images

  def get_image_block(self):
    """ Loops through all images belonging to a Site, flattens pixel values,
    and returns a pandas DataFrame for all values

    If image dimensions are inconsistent, the images will be rescaled to the
    largest image size.

    :return pandas DataFrame of flattened pixel values, along with image
    coordinates and Site name
    """
    imgs = {i.name if i.maskName is None else i.maskName:i.get_image()
            for i in self.images}
    maxRes = max((i.shape for i in imgs.values()))
    rr, cc = np.mgrid[0:maxRes[0], 0:maxRes[1]]
    imgBlock = {'imgRow': rr.astype(np.uint16).flatten(),
                "imgCol": cc.astype(np.uint16).flatten()}
    imgBlock.update({k:mh.imresize(v, maxRes, order=3).flatten()
                     for k,v in imgs.items()})
    df = pd.DataFrame(imgBlock)

    df['site'] = self.name
    return df

  def get_mask_names(self):
    """ Gets the set of mask names for each image

    :return set of mask names as strings
    """
    return set(i.maskName for i in self.images)


  def __str__(self):
    return self.name + "-->" + ",".join(str(x) for x in self.images)


class Image():
  def __init__(self, name, path, img_type, maskName=None):
    """ Creates a new channel of information, i.e. BSE or EDX image/text file

    :param name: unique identifier of this Image within the scope of the parent
    Site
    :param path: filepath
    :param img_type: tag, e.g. 'EDX', 'BSE' or 'MASK'
    """
    self.name = name.capitalize()
    self.path = path
    self.img_type = img_type.upper()
    # Discern image format
    _, imgFormat = os.path.splitext(self.path)
    self.imgFormat = imgFormat.lower()
    self.maskName = maskName

  def get_image(self):
    """ Reads an image from file depending on the file format

    :return a numpy nd-array of the image, or None
    """
    # TODO: In the future, the imgType may be used for specifying data type
    if self.imgFormat == ".tsv":
      return np.loadtxt(self.path, dtype=np.float64)
    else:
      try:
        im = io.imread(self.path, flatten=True)
        if self.maskName is not None:
          return im.astype(np.bool)
        else:
          return im
      except:
        print("Error: image file %s could not be loaded" % self.path)
        raise

  def __str__(self):
    return self.name


def get_material_list(dataDict):
  """ Instantiates Image, Site, and Material objects given a nested dictionary

  :param dataDict: a nested dictionary conforming to the JSON-like structure
  given in
  :return a list of Materials, Sites, or Images
  """
  keys = dataDict.keys()
  if "images" in keys:
    return [Image(i["name"],
                  i["path"],
                  i["type"],
                  i.get("maskName"))
            for i in dataDict.get("images")]
  if "sites" in keys:
    return [Site(v["name"],
                 v["path"],
                 get_material_list(v))
            for v in dataDict.get("sites")]
  if "materials" in keys:
    return [Material(v["name"],
                     v["path"],
                     get_material_list(v))
            for v in dataDict.get("materials")]


def write_db(ctrlFilePath, storePath):
  """ Creates a new dataframe from the control file and writes it to disk

  :param ctrlFilePath: JSON control file specifying how to build the DataFrame
  :param storePath: string specifying absolute path to the HDF5 store to be
  created
  """
  ctrlDict = get_input_json(ctrlFilePath)
  matList = get_material_list(ctrlDict)
  with stopwatch('writing to HDF5 store'):
    with pd.HDFStore(storePath,'w') as store:
      for mat in matList:
        chunk = mat.get_dataframe()
        print(chunk.info())
        print(chunk.head())
        store.append(mat.name, chunk, data_columns=True, index=True)


def overwrite_db():
  make_input_json()
  write_db('../input_data/ctrl_00.json','../output/store.h5')


def explore_db():
  df = pd.read_hdf("../output/store.h5",
                   "FAF",
                   where=[
                     "site='soi_004'",
                   ])

  print(df.info())
  print(df.head())
  print(df.count())


if __name__ == '__main__':
  overwrite_db()
  # explore_db()
