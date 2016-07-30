# -*- coding: utf-8 -*-
from semclassify import helpers as h
import numpy as np
import pandas as pd
from skimage import io
from itertools import groupby
import json
import os

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

  def get_db(self):
    with h.stopwatch("creating DataFrame for material %s" % self.name):
      df = pd.DataFrame()
      for s in self.sites:
        df = df.append(s.get_image_block())
      df['material'] = self.name
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
    imgBlock.update({k:np.resize(v, maxRes).flatten()
                     for k,v in imgs.items()})
    df = pd.DataFrame(imgBlock)

    df['site'] = self.name
    return df

  def __str__(self):
    return self.name + "-->" + ",".join(str(x) for x in self.images)


class Image():
  def __init__(self, name, path, imgType, maskName=None):
    """ Creates a new channel of information, i.e. BSE or EDX image/text file

    :param name: unique identifier of this Image within the scope of the parent
    Site
    :param path: filepath
    :param imgType: tag, e.g. 'EDX', 'BSE' or 'MASK'
    """
    self.name = name.capitalize()
    self.path = path
    self.imgType = imgType.upper()
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

        im = io.imread(self.path, flatten=False)
        if self.maskName is not None:
          return im
        else:
          return im.astype(np.bool)

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
  ctrlDict = h.get_input_json(ctrlFilePath)
  matList = get_material_list(ctrlDict)
  with h.stopwatch('writing to HDF5 store'):
    with pd.HDFStore(storePath,'w') as store:
      for mat in matList:
        chunk = mat.get_db()
        print(chunk.info())
        store.append('df', chunk, data_columns=True, index=True)

def query_store(store_path, where=None, columns=None):
  with h.stopwatch("querying store"):
    with pd.HDFStore(store_path,'r') as store:
      return store.select('df',
                          where=where,
                          columns=columns)


def actions(choice='read'):
  ctrlFilePath = "../input_data/ctrl_00.json"
  storePath = "../output/store.h5"

  if choice == 'read':
    return query_store(storePath,
                       where='material=="BFS" & '
                             'site="soi_001" & '
                             'BFS!=0',
                       columns=['Al', 'BSE'])
  elif choice == 'write':
    write_db(ctrlFilePath, storePath)
  else:
    raise NameError("Unrecognized option.")


if __name__ == '__main__':
  # print(actions('write'))
  print(actions('read'))
