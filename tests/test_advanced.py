# -*- coding: utf-8 -*-

from .context import semclassify as sc

import unittest


class AdvancedTestSuite(unittest.TestCase):
  """Advanced test cases."""

  def test_thoughts(self):
    with sc.stopwatch("testing"):
      pass


if __name__ == '__main__':
  unittest.main()