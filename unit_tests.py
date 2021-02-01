# import os
#
# if "CI" in os.environ:
#     os.environ["NUMBA_ENABLE_CUDASIM"] = "1"

import unittest

import fractal_core as core


class CoreTest(unittest.TestCase):
    @staticmethod
    def test_mandel():
        core.mandel(-2, 1, -1, 1, (1280, 720), 100)

    @staticmethod
    def test_mandel_color():
        core.mandel_color(-2, 1, -1, 1, (1280, 720), 100)

    @staticmethod
    def test_julia():
        core.julia(-1, 1, -1, 1, (1280, 720), 100, 0)

    @staticmethod
    def test_julia_color():
        core.julia_color(-2, 1, -1, 1, (1280, 720), 100, 0)

    @staticmethod
    def test_carpet():
        core.carpet(3)
