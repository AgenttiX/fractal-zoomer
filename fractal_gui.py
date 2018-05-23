# Created by Mika Mäki, 2018
# for Tampere University of Technology course
# RAK-19006 Python 3 for scientific computing

import fractal_core as frac

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from pyqtgraph.Qt import QtWidgets

import numba
import numpy as np
import scipy.misc

import time
import copy


imsave_accel = numba.jit(scipy.misc.imsave)


class FractalGUI:
    def_res_x = 4096
    def_res_y = 2160
    def_x_min = -2.0
    def_x_max = 1.0
    def_y_min = -1.0
    def_y_max = 1.0
    def_c_real = 0
    def_c_imag = 0
    def_iter = 200
    def_frames = 100

    def __init__(self):
        app = pg.mkQApp()
        pg.setConfigOptions(antialias=True)

        win = QtGui.QMainWindow()
        win.resize(1200, 700)

        self.__imv = pg.ImageView()
        win.setCentralWidget(self.__imv)
        win.setWindowTitle("Fractal view")
        self.__imv.getView().invertY(False)     # Disable the default y axis inversion

        win2 = QtGui.QWidget()
        win2.setWindowTitle("Fractal controls")
        win2_layout = QtGui.QGridLayout()
        win2.setLayout(win2_layout)

        labels = [
            "fractal",
            "x resolution",
            "y resolution",
            "x min",
            "x max",
            "y min",
            "y max",
            "c real",
            "c imag",
            "iterations"
        ]

        for i, text in enumerate(labels):
            label = QtWidgets.QLabel()
            label.setText(text)
            win2_layout.addWidget(label, i, 0)

        self.__res_x = FractalGUI.def_res_x
        self.__res_y = FractalGUI.def_res_y

        self.__x_min = FractalGUI.def_x_min
        self.__x_max = FractalGUI.def_x_max
        self.__y_min = FractalGUI.def_y_min
        self.__y_max = FractalGUI.def_y_max

        self.__c_real = FractalGUI.def_c_real
        self.__c_imag = FractalGUI.def_c_imag
        self.__iter_max = FractalGUI.def_iter

        self.__start_x_min = None
        self.__start_x_max = None
        self.__start_y_min = None
        self.__start_y_max = None
        self.__start_c_real = None
        self.__start_c_imag = None
        self.__start_iter_max = None

        self.__end_x_min = None
        self.__end_x_max = None
        self.__end_y_min = None
        self.__end_y_max = None
        self.__end_c_real = None
        self.__end_c_imag = None
        self.__end_iter_max = None

        self.__image = None

        self.__input_frac = QtWidgets.QComboBox()
        self.__input_frac.addItem("Mandelbrot", "mandel")
        self.__input_frac.addItem("Mandelbrot colored", "mandel-color")
        self.__input_frac.addItem("Julia", "julia")
        self.__input_frac.addItem("Sierpinski carpet", "carpet")
        win2_layout.addWidget(self.__input_frac, 0, 1)

        self.__input_res_x = pg.SpinBox(value=self.__res_x, int=True, dec=True, minStep=1, step=1)
        self.__input_res_y = pg.SpinBox(value=self.__res_y, int=True, dec=True, minStep=1, step=1)
        win2_layout.addWidget(self.__input_res_x, 1, 1)
        win2_layout.addWidget(self.__input_res_y, 2, 1)

        self.__input_x_min = pg.SpinBox(value=self.__x_min, dec=True)
        self.__input_x_max = pg.SpinBox(value=self.__x_max, dec=True)
        self.__input_y_min = pg.SpinBox(value=self.__y_min, dec=True)
        self.__input_y_max = pg.SpinBox(value=self.__y_max, dec=True)
        win2_layout.addWidget(self.__input_x_min, 3, 1)
        win2_layout.addWidget(self.__input_x_max, 4, 1)
        win2_layout.addWidget(self.__input_y_min, 5, 1)
        win2_layout.addWidget(self.__input_y_max, 6, 1)

        self.__input_c_real = pg.SpinBox(value=self.__c_real, dec=True)
        self.__input_c_imag = pg.SpinBox(value=self.__c_imag, dec=True)
        win2_layout.addWidget(self.__input_c_real, 7, 1)
        win2_layout.addWidget(self.__input_c_imag, 8, 1)

        self.__input_iter = pg.SpinBox(value=self.__iter_max, int=True, dec=True)
        win2_layout.addWidget(self.__input_iter, 9, 1)

        self.__renderButton = QtWidgets.QPushButton("Render")
        win2_layout.addWidget(self.__renderButton, 10, 1)
        self.__renderButton.clicked.connect(self.render)

        self.__zoomButton = QtWidgets.QPushButton("Zoom")
        win2_layout.addWidget(self.__zoomButton, 11, 1)
        self.__zoomButton.clicked.connect(self.zoom)

        self.__saveButton = QtWidgets.QPushButton("Save")
        win2_layout.addWidget(self.__saveButton, 11, 1)
        self.__saveButton.clicked.connect(self.save)

        self.__resetButton = QtWidgets.QPushButton("Reset")
        win2_layout.addWidget(self.__resetButton, 12, 1)
        self.__resetButton.clicked.connect(self.reset)

        label = QtWidgets.QLabel()
        label.setText("frames")
        win2_layout.addWidget(label, 0, 2)

        self.__input_frames = pg.SpinBox(value=FractalGUI.def_frames, dec=True, int=True, minStep=1, step=1)
        win2_layout.addWidget(self.__input_frames, 0, 3)

        self.__startButton = QtWidgets.QPushButton("Set start frame")
        self.__endButton = QtWidgets.QPushButton("Set end frame")
        win2_layout.addWidget(self.__startButton, 1, 2)
        win2_layout.addWidget(self.__endButton, 1, 3)
        self.__startButton.clicked.connect(self.set_start)
        self.__endButton.clicked.connect(self.set_end)

        self.__animeButton = QtWidgets.QPushButton("Render animation")
        self.__animeButton.clicked.connect(self.animate)
        win2_layout.addWidget(self.__animeButton, 2, 2)

        self.__testButton = QtWidgets.QPushButton("Test")
        self.__testButton.clicked.connect(self.test)
        win2_layout.addWidget(self.__testButton, 2, 3)

        win.show()
        win2.show()

        # Note: PyQtGraph is prone to crashing on exit
        # (mysterious segfaults etc., especially when used along with other libraries)
        # See this for documentation:
        # http://www.pyqtgraph.org/documentation/functions.html#pyqtgraph.exit
        app.exec_()
        # pg.exit()

    def render(self):
        self.__res_x = self.__input_res_x.value()
        self.__res_y = self.__input_res_y.value()
        self.__x_min = self.__input_x_min.value()
        self.__x_max = self.__input_x_max.value()
        self.__y_min = self.__input_y_min.value()
        self.__y_max = self.__input_y_max.value()
        self.__iter_max = self.__input_iter.value()
        self.__c_real = self.__input_c_real.value()
        self.__c_imag = self.__input_c_imag.value()

        self.render_engine()

    def render_engine(self):
        selection = self.__input_frac.currentData()

        color = False

        start_time = time.process_time()

        if selection == "mandel":
            self.__image = frac.mandel(
                self.__x_min,
                self.__x_max,
                self.__y_min,
                self.__y_max,
                (self.__res_y, self.__res_x),
                self.__iter_max
            )
        elif selection == "mandel-color":
            self.__image = frac.mandel_color(
                self.__x_min,
                self.__x_max,
                self.__y_min,
                self.__y_max,
                (self.__res_y, self.__res_x),
                self.__iter_max
            )
            color = True
        elif selection == "julia":
            self.__image = frac.julia(
                self.__x_min,
                self.__x_max,
                self.__y_min,
                self.__y_max,
                (self.__res_y, self.__res_x),
                self.__iter_max,
                complex(self.__c_real, self.__c_imag)
            )
        elif selection == "carpet":
            self.__image = frac.carpet(self.__res_x)
        else:
            raise RuntimeError("Render function received an invalid option from QComboBox")

        print("Render time", time.process_time() - start_time)

        # coloring_start_time = time.process_time()
        # colored = frac.color(self.__image, self.__iter_max)
        # print("Coloring time", time.process_time() - coloring_start_time)

        drawing_start_time = time.process_time()

        if color:
            self.__imv.setImage(self.__image, axes={"x": 1, "y": 0, "c": 2})
        else:
            self.__imv.setImage(self.__image, axes={"x": 1, "y": 0})

        print("Drawing time", time.process_time() - drawing_start_time)

    def zoom(self):
        pass

    def save(self, filename: str = None):
        start_time = time.process_time()
        if filename is None:
            imsave_accel("/dev/shm/test.png", self.__image)
        else:
            imsave_accel("/dev/shm/" + filename + ".png", self.__image)
        print("Saving time", time.process_time() - start_time)

    def reset(self):
        self.__res_x = FractalGUI.def_res_x
        self.__res_y = FractalGUI.def_res_y
        self.__x_min = FractalGUI.def_x_min
        self.__x_max = FractalGUI.def_x_max
        self.__y_min = FractalGUI.def_y_min
        self.__y_max = FractalGUI.def_y_max
        self.__c_real = FractalGUI.def_c_real
        self.__c_imag = FractalGUI.def_c_imag
        self.__iter_max = FractalGUI.def_iter

        self.__input_res_x.setValue(self.__res_x)
        self.__input_res_x.setValue(self.__res_y)
        self.__input_x_min.setValue(self.__x_min)
        self.__input_x_max.setValue(self.__x_max)
        self.__input_y_min.setValue(self.__y_min)
        self.__input_y_max.setValue(self.__y_max)

    def set_start(self):
        self.__start_x_min = copy.deepcopy(self.__x_min)
        self.__start_x_max = copy.deepcopy(self.__x_max)
        self.__start_y_min = copy.deepcopy(self.__y_min)
        self.__start_y_max = copy.deepcopy(self.__y_max)
        self.__start_c_real = copy.deepcopy(self.__c_real)
        self.__start_c_imag = copy.deepcopy(self.__c_imag)
        self.__start_iter_max = copy.deepcopy(self.__iter_max)
    
    def set_end(self):
        self.__end_x_min = copy.deepcopy(self.__x_min)
        self.__end_x_max = copy.deepcopy(self.__x_max)
        self.__end_y_min = copy.deepcopy(self.__y_min)
        self.__end_y_max = copy.deepcopy(self.__y_max)
        self.__end_c_real = copy.deepcopy(self.__c_real)
        self.__end_c_imag = copy.deepcopy(self.__c_imag)
        self.__end_iter_max = copy.deepcopy(self.__iter_max)

    def animate(self):
        frames = self.__input_frames.value()

        x_min = np.linspace(self.__start_x_min, self.__end_x_min, frames)
        x_max = np.linspace(self.__start_x_max, self.__end_x_max, frames)
        y_min = np.linspace(self.__start_y_min, self.__end_y_min, frames)
        y_max = np.linspace(self.__start_y_max, self.__end_y_max, frames)
        c_real = np.linspace(self.__start_c_real, self.__end_c_real, frames)
        c_imag = np.linspace(self.__start_c_imag, self.__end_c_imag, frames)
        iter_max = np.linspace(self.__start_iter_max, self.__end_iter_max, frames, dtype=np.int)

        for i in range(frames):
            self.__x_min = x_min[i]
            self.__x_max = x_max[i]
            self.__y_min = y_min[i]
            self.__y_max = y_max[i]
            self.__c_real = c_real[i]
            self.__c_imag = c_imag[i]
            self.__iter_max = iter_max[i]

            self.render_engine()
            self.save(str(i))

    def test(self):
        print(self.__imv.getView().viewRect())


if __name__ == "__main__":
    FractalGUI()
