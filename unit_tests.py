import fractal_core as core

import unittest


class CoreTest(unittest.TestCase):
    def test_mandel(self):
        core.mandel(-2, 1, -1, 1, (1280, 720), 100)

    def test_mandel_color(self):
        core.mandel_color(-2, 1, -1, 1, (1280, 720), 100)

    def test_julia(self):
        core.julia(-1, 1, -1, 1, (1280, 720), 100, 0)

    def test_julia_color(self):
        core.julia_color(-2, 1, -1, 1, (1280, 720), 100, 0)

    def test_carpet(self):
        core.carpet(3)
