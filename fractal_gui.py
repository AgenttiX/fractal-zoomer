# Created by Mika Mäki, 2018
# for Tampere University of Technology course
# RAK-19006 Python 3 for scientific computing

# TODO: Numba is broken with CUDA 11.2 until Numba 0.53.0
# TODO: Test with 0.53.0 whether this has been fixed
# The produced error is:
# numba.cuda.cudadrv.error.NvvmError: Failed to compile
# <unnamed> (54, 19): parse expected comma after load's type
# NVVM_ERROR_COMPILATION

import fractal_core as frac

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

# import imageio
import numba
import numpy as np
import scipy.misc

import glob
import multiprocessing as mp
import os
import os.path
import subprocess
import time


# An attempt to speed up the PNG saving
# (The saving is parallelized below, so the nogil is pretty much unnecessary)
# TODO: test and enable this
# save_accel = numba.jit(imageio.imwrite, nogil=True)
# The imsave function has been removed in SciPy 1.2.0
# https://stackoverflow.com/questions/49319841/where-is-imsave-in-scipy-1-0-0
save_accel = numba.jit(scipy.misc.imsave, nogil=True)


def save_process(q: mp.JoinableQueue):
    """
    A worker for saving PNG images
    Saving the resulting image was the slowest step in the rendering process, so it's parallelized
    :param q:
    :return:
    """
    while True:
        start_time = time.process_time()
        path, image = q.get()   # This call is blocking, so the while loop doesn't run on its own
        print("Saving", path)
        # save_accel(path, image)
        scipy.misc.imsave(path, np.flipud(image))
        q.task_done()
        print("Saving time", time.process_time() - start_time)


class FractalGUI:
    """
    The GUI class of the fractal software
    """

    # Default parameters

    def_res_x = 1920
    def_res_y = 1080

    # The resulting 4K video would be highly resource intensive to play
    # def_res_x = 3840
    # def_res_y = 2160

    def_x_min = -2.0
    def_x_max = 1.0
    def_y_min = -1.0
    def_y_max = 1.0
    def_c_real = 0
    def_c_imag = 0
    def_iter = 200
    def_frames = 100
    def_fps = 30

    def_path = "/dev/shm/fract"

    def __init__(self):
        if not os.path.exists(FractalGUI.def_path):
            os.mkdir(FractalGUI.def_path)

        # Create workers for image saving
        mp.set_start_method("spawn")
        self.__save_queue = mp.JoinableQueue()
        self.__save_pool = mp.Pool(initializer=save_process, initargs=(self.__save_queue,))

        # Initialise PyQtGraph
        app = pg.mkQApp()
        pg.setConfigOptions(antialias=True)

        win = QtWidgets.QMainWindow()
        win.resize(1200, 700)

        # Couldn't get multiprocess graphics rendering to work
        # graphics_view = pyqtgraph.widgets.RemoteGraphicsView.RemoteGraphicsView()   # useOpenGL=True speeds the rendering
        # self.__imv = pg.ImageView(view=graphics_view)

        self.__imv = pg.ImageView()
        win.setCentralWidget(self.__imv)
        win.setWindowTitle("Fractal view")
        self.__imv.getView().invertY(False)     # Disable the default y axis inversion

        win2 = QtWidgets.QWidget()
        win2.setWindowTitle("Fractal controls")
        win2_layout = QtWidgets.QGridLayout()
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
            frame_label = QtWidgets.QLabel()
            frame_label.setText(text)
            win2_layout.addWidget(frame_label, i, 0)

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

        self.__image: np.ndarray = None

        self.__input_frac = QtWidgets.QComboBox()
        self.__input_frac.addItem("Mandelbrot colored", "mandel-color")
        self.__input_frac.addItem("Mandelbrot grayscale", "mandel")
        self.__input_frac.addItem("Julia colored", "julia-color")
        self.__input_frac.addItem("Julia grayscale", "julia")
        # This works but would require different parameter entries in the GUI
        # self.__input_frac.addItem("Sierpinski carpet", "carpet")
        win2_layout.addWidget(self.__input_frac, 0, 1)

        self.__input_res_x = pg.SpinBox(value=self.__res_x, int=True, dec=True, minStep=1, step=1, min=10)
        self.__input_res_y = pg.SpinBox(value=self.__res_y, int=True, dec=True, minStep=1, step=1, min=10)
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

        self.__input_c_real = pg.SpinBox(value=self.__c_real, dec=True, minStep=0)
        self.__input_c_imag = pg.SpinBox(value=self.__c_imag, dec=True, minStep=0)
        win2_layout.addWidget(self.__input_c_real, 7, 1)
        win2_layout.addWidget(self.__input_c_imag, 8, 1)

        self.__input_iter = pg.SpinBox(value=self.__iter_max, int=True, dec=True, min=1)
        win2_layout.addWidget(self.__input_iter, 9, 1)

        self.__renderButton = QtWidgets.QPushButton("Render")
        win2_layout.addWidget(self.__renderButton, 10, 1)
        self.__renderButton.clicked.connect(self.render)

        self.__zoomButton = QtWidgets.QPushButton("Zoom")
        win2_layout.addWidget(self.__zoomButton, 11, 1)
        self.__zoomButton.clicked.connect(self.zoom)

        self.__saveButton = QtWidgets.QPushButton("Save")
        win2_layout.addWidget(self.__saveButton, 12, 1)
        self.__saveButton.clicked.connect(self.save)

        self.__resetButton = QtWidgets.QPushButton("Reset")
        win2_layout.addWidget(self.__resetButton, 13, 1)
        self.__resetButton.clicked.connect(self.reset)

        frame_label = QtWidgets.QLabel()
        frame_label.setText("frames")
        win2_layout.addWidget(frame_label, 0, 2)

        fps_label = QtWidgets.QLabel()
        fps_label.setText("FPS")
        win2_layout.addWidget(fps_label, 1, 2)

        self.__input_frames = pg.SpinBox(value=FractalGUI.def_frames, dec=True, int=True, minStep=1, step=1, min=1)
        win2_layout.addWidget(self.__input_frames, 0, 3)

        self.__input_fps = pg.SpinBox(value=FractalGUI.def_fps, dec=True, int=True, minStep=1, step=1, min=1)
        win2_layout.addWidget(self.__input_fps, 1, 3)

        self.__startButton = QtWidgets.QPushButton("Set start frame")
        self.__endButton = QtWidgets.QPushButton("Set end frame")
        win2_layout.addWidget(self.__startButton, 2, 2)
        win2_layout.addWidget(self.__endButton, 2, 3)
        self.__startButton.clicked.connect(self.set_start)
        self.__endButton.clicked.connect(self.set_end)

        self.__animeButton = QtWidgets.QPushButton("Render animation")
        self.__animeButton.clicked.connect(self.animate)
        win2_layout.addWidget(self.__animeButton, 3, 2)

        # self.__testButton = QtWidgets.QPushButton("Test")
        # self.__testButton.clicked.connect(self.test)
        # win2_layout.addWidget(self.__testButton, 3, 3)

        self.__renderLabel = QtWidgets.QLabel()
        self.__renderLabel.setText("")
        win2_layout.addWidget(self.__renderLabel, 4, 2)

        win.show()
        win2.show()

        self.render()

        # Note: PyQtGraph is prone to crashing on exit
        # (mysterious segfaults etc., especially when used along with other libraries)
        # This is not caused by improper usage but bugs in the library itself
        #  Please see this for documentation:
        # http://www.pyqtgraph.org/documentation/functions.html#pyqtgraph.exit
        app.exec_()

        # This can prevent some of the errors
        # pg.exit()

    def render(self):
        """
        Render a new frame based on the GUI parameters. Uses render_engine() for the actual work.
        :return: -
        """
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
        """
        Render a new frame without updating parameter values from the GUI
        :return: -
        """

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
        elif selection == "julia-color":
            self.__image = frac.julia_color(
                self.__x_min,
                self.__x_max,
                self.__y_min,
                self.__y_max,
                (self.__res_y, self.__res_x),
                self.__iter_max,
                complex(self.__c_real, self.__c_imag)
            )
            color = True
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
        """
        Render the fractal as zoomed in the GUI
        :return: -
        """
        rect = self.__imv.getView().viewRect()

        width = self.__x_max - self.__x_min
        height = self.__y_max - self.__y_min

        zoom_factor = rect.width() / self.__res_x

        px_center = rect.center()
        center_x = self.__x_min + px_center.x() / self.__res_x * width
        center_y = self.__y_min + px_center.y() / self.__res_y * height

        self.__x_max = center_x + zoom_factor * width * 0.5
        self.__x_min = center_x - zoom_factor * width * 0.5
        self.__y_max = center_y + zoom_factor * height * 0.5
        self.__y_min = center_y - zoom_factor * height * 0.5

        self.__input_x_max.setValue(self.__x_max)
        self.__input_x_min.setValue(self.__x_min)
        self.__input_y_max.setValue(self.__y_max)
        self.__input_y_min.setValue(self.__y_min)

        self.render()

    def save(self, filename: str = None):
        """
        Save the fractal to a file
        :param filename: file name
        :return: -
        """
        start_time = time.process_time()
        if type(filename) is not str:
            save_accel(os.path.join(FractalGUI.def_path, "test.png"), np.flipud(self.__image))
        else:
            save_accel(os.path.join(FractalGUI.def_path, filename) + ".png", np.flipud(self.__image))
        print("Saving time", time.process_time() - start_time)

    def reset(self):
        """
        Reset the fractal to default parameters
        :return: -
        """
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
        self.__input_res_y.setValue(self.__res_y)
        self.__input_x_min.setValue(self.__x_min)
        self.__input_x_max.setValue(self.__x_max)
        self.__input_y_min.setValue(self.__y_min)
        self.__input_y_max.setValue(self.__y_max)
        self.__input_c_real.setValue(self.__c_real)
        self.__input_c_imag.setValue(self.__c_imag)
        self.__input_iter.setValue(self.__iter_max)

        self.render_engine()

    def set_start(self):
        """
        Set parameters for the first frame of the animation
        :return: -
        """
        self.__start_x_min = self.__x_min
        self.__start_x_max = self.__x_max
        self.__start_y_min = self.__y_min
        self.__start_y_max = self.__y_max
        self.__start_c_real = self.__c_real
        self.__start_c_imag = self.__c_imag
        self.__start_iter_max = self.__iter_max
    
    def set_end(self):
        """
        Set parameters for the last frame of the animation
        :return: -
        """
        self.__end_x_min = self.__x_min
        self.__end_x_max = self.__x_max
        self.__end_y_min = self.__y_min
        self.__end_y_max = self.__y_max
        self.__end_c_real = self.__c_real
        self.__end_c_imag = self.__c_imag
        self.__end_iter_max = self.__iter_max

    def animate(self):
        """
        Render and save the animation
        :return: -
        """

        can_start = True
        if self.__start_x_max is None:
            self.print("Start frame has not been set")
            can_start = False
        if self.__end_x_max is None:
            self.print("End frame has not been set")
            can_start = False

        if not can_start:
            return

        self.print("Animating")
        time.sleep(0.1)
        start_time = time.process_time()

        frames = self.__input_frames.value()
        fps = self.__input_fps.value()

        frame_number_len = len(str(frames-1))

        x_min = np.linspace(self.__start_x_min, self.__end_x_min, frames)
        x_max = np.linspace(self.__start_x_max, self.__end_x_max, frames)
        y_min = np.linspace(self.__start_y_min, self.__end_y_min, frames)
        y_max = np.linspace(self.__start_y_max, self.__end_y_max, frames)
        c_real = np.linspace(self.__start_c_real, self.__end_c_real, frames)
        c_imag = np.linspace(self.__start_c_imag, self.__end_c_imag, frames)
        iter_max = np.linspace(self.__start_iter_max, self.__end_iter_max, frames, dtype=np.int)

        regex = glob.glob(os.path.join(FractalGUI.def_path, "frame") + "*.png")
        for file in regex:
            os.remove(file)

        out_path = os.path.join(FractalGUI.def_path, "out.mp4")
        if os.path.exists(out_path):
            os.remove(out_path)

        for i in range(frames):
            self.print("Rendering " + str(i) + "/" + str(frames))

            self.__x_min = x_min[i]
            self.__x_max = x_max[i]
            self.__y_min = y_min[i]
            self.__y_max = y_max[i]
            self.__c_real = c_real[i]
            self.__c_imag = c_imag[i]
            self.__iter_max = iter_max[i]

            self.render_engine()
            # self.save("frame_" + str(i).zfill(frame_number_len))
            file_name = os.path.join(FractalGUI.def_path, "frame_" + str(i).zfill(frame_number_len) + ".png")
            self.__save_queue.put((file_name, self.__image))

        self.print("Saving frames")
        self.__save_queue.join()

        self.print("Rendering video")
        subprocess.run([
            "ffmpeg", "-r",
            str(fps),
            "-pattern_type",
            "glob",
            "-i",
            os.path.join(FractalGUI.def_path, "frame_*.png"),
            os.path.join(FractalGUI.def_path, "out.mp4")
             ])
        self.print("Ready")

        total_time = time.process_time() - start_time
        print("Total animation time:", total_time)
        print("Time per frame", total_time / frames)

    # def test(self):
    #     print(self.__imv.getView().viewRect())

    def print(self, text: str):
        self.__renderLabel.setText(text)
        print(text)


if __name__ == "__main__":
    FractalGUI()
