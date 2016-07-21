# -*- coding: utf-8 -*-
from helpers import *
import numpy as np
from scipy.ndimage import imread
import pandas as pd


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
    with stopwatch("Creating DataFrame for material %s" % self.name):
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
    imgs = {i.name:i.get_image() for i in self.images}
    maxRes = max((i.shape for i in imgs.itervalues()))
    rr, cc = np.mgrid[0:maxRes[0], 0:maxRes[1]]
    imgBlock = {'imgRow': rr.flatten(), "imgCol": cc.flatten()}
    imgBlock.update({k:np.resize(v, maxRes).flatten()
                     for k,v in imgs.iteritems()})
    df = pd.DataFrame(imgBlock)
    df['site'] = self.name
    return pd.DataFrame(imgBlock)

  def __str__(self):
    return self.name + "-->" + ",".join(str(x) for x in self.images)


class Image():
  def __init__(self, name, path, imgType):
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

  def get_image(self):
    """ Reads an image from file depending on the file format

    :return a numpy nd-array of the image, or None
    """
    if self.imgFormat == ".tsv":
      return np.loadtxt(self.path)
    elif self.imgFormat == ".tif" or self.imgFormat == ".tiff":
      return imread(self.path)
    else:
      return None

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
    return [Image(i.get("name"), i.get("path"), i.get("type"))
            for i in dataDict.get("images")]
  if "sites" in keys:
    return [Site(v.get("name"), v.get("path"), get_material_list(v))
            for v in dataDict.get("sites")]
  if "materials" in keys:
    return [Material(v.get("name"), v.get("path"), get_material_list(v))
            for v in dataDict.get("materials")]


if __name__ == '__main__':

  matDict = make_input_json()
  # print json.dumps(matDict, indent=4, sort_keys=False)

  matList = get_material_list(matDict)
  # for i in matList: print i

  foo = \
    Image("Si","/Users/joshuaarnold/Documents/Papers/VU_SEM/analysis/SEM-EDX" \
            "-DATA/BFS/soi_013/TSV-TIFF/Si.tsv", "EDX")

  m = matList[0].get_db()
  print m.columns
  print m
