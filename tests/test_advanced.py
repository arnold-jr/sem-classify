# -*- coding: utf-8 -*-

from .context import sem_build_db

import unittest


class AdvancedTestSuite(unittest.TestCase):
    """Advanced test cases."""

    def test_thoughts(self):
        sem_build_db.hmm()


if __name__ == '__main__':
    unittest.main()