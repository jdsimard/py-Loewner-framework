import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import loewner_framework as lf
import loewner_framework.linear_daes as ld


Lambda = np.diag([0.0, complex(imag=1.0), complex(imag=-1.0)])
R = np.array([[1, 1, 1]])
W = np.array([[complex(real=5.0), complex(real=0.5, imag=-0.1), complex(real=0.5, imag=0.1)]])
rtd = lf.RightTangentialData(Lambda, R, W)

M = np.diag([1.0, complex(imag=10.0), complex(imag=-10.0)])
L = np.array([  [1],
                [1],
                [1]])
V = np.array([  [1],
                [complex(real=1.0, imag=-0.5)],
                [complex(real=1.0, imag=0.5)]])
ltd = lf.LeftTangentialData(M, L, V)

loewnertd = lf.LoewnerTangentialData(rtd, ltd)
interpolant_builder = lf.InterpolantFactory(loewnertd)
interpolant = interpolant_builder.minimal_order(label="Interpolant")


bode_plot = ld.BodePlot()
bode_plot.add_system(interpolant, color='b')

bode_plot.add_data_tick(at_frequency=10**-2, pin_to_system_num=0, pin_to_output_num=1, pin_to_input_num=1, markersize=12, color="red")
bode_plot.add_data_tick(at_frequency=10**0, pin_to_system_num=0, pin_to_output_num=1, pin_to_input_num=1, markersize=12, color="red")
bode_plot.add_data_tick(at_frequency=10**1, pin_to_system_num=0, pin_to_output_num=1, pin_to_input_num=1, markersize=12, color="red")

bode_plot.show(w_start=-2, w_end=2, w_num_points=10000)


print(interpolant.E, interpolant.A, interpolant.B, interpolant.C, interpolant.D)
print(interpolant.tf(complex(real=0.0, imag=1.0)))