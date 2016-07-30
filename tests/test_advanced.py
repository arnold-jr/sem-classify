# -*- coding: utf-8 -*-

from .context import semclassify as sc
from sklearn import datasets
import unittest


class AdvancedTestSuite(unittest.TestCase):
  """Advanced test cases."""

  def test_classifiers(self):
    with sc.helpers.stopwatch("testing"):
      sm = sc.classify.SuperModel(['Al','Si'], ['BFS'])
      self.assertIsInstance(sm,sc.classify.SuperModel)
      print("hello")


if __name__ == '__main__':
  unittest.main()