# -*- coding: utf-8 -*-

from .context import semclassify as sc

import unittest

class BasicTestSuite(unittest.TestCase):
  """Basic test cases."""
  def setUp(self):
    self.image = sc.Image("Si",
               "/Users/joshuaarnold/Documents/Papers/VU_SEM/analysis/SEM-EDX"
                   "-DATA/BFS/soi_013/TSV-TIFF/Si.tsv",
                "EDX",
                None)
    self.images = [
      sc.Image("Al",
              "/Users/joshuaarnold/Documents/Papers/VU_SEM/analysis/"
                  "SEM-EDX-DATA/BFS/soi_001/TSV-TIFF/Al.tsv",
              "EDX"),
      sc.Image("BFS",
               "/Users/joshuaarnold/Documents/Papers/VU_SEM/analysis/"
                  "SEM-EDX-DATA/BFS/soi_001/TSV-TIFF/BFS.tif",
               'BSE',
               maskName='BFS')
    ]

  def test_image_class(self):
    self.assertIsInstance(self.image,sc.Image, "The object is not an instance "
                                        "of class Image")

    for i in self.images:
      self.assertIsInstance(i,sc.Image, "The object is not an instance of class"
                                        "image")


if __name__ == '__main__':
  unittest.main()