import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import scipy.linalg as spla

import loewner_framework as lf
import loewner_framework.linear_daes as ld


n, m, p = 10, 3, 2
E = np.eye(n)
A = np.random.sample((n,n))
B = np.random.sample((n,m))
C = np.random.sample((p,n))
D = np.random.sample((p,m))
system = ld.LinearDAE(A, B, C, D, label="System")
system.label += f" (n = {system.n_r})"






Lambda = np.array([ [0, 0, 0],
                    [0, 0, 1],
                    [0, -1, 0]])
R = np.array([  [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1]])
rtd = lf.RightTangentialData(Lambda, R, system)

M = np.array([  [1, 0, 0],
                [0, 0, -100],
                [0, 100, 0]])
L = np.array([  [1, 0],
                [0, 1],
                [1, 1]])
ltd = lf.LeftTangentialData(M, L, system)

loewnertd = lf.LoewnerTangentialData(rtd, ltd)

interpolant_builder = lf.InterpolantFactory(loewnertd)

desired_poles = np.arange(-6, 0, 1) * 1
poleplaced_interp = interpolant_builder.double_order_pole_placed(desired_poles, D=None, label="placed")
#poleplaced_interp = interpolant_builder.double_order_pole_placed(desired_poles, D=np.random.random_sample((p, m)), label="Pole-Placed Interpolant")
print(poleplaced_interp)
print(spla.eigvals(poleplaced_interp.A, poleplaced_interp.E))
print(spla.eigvals(np.linalg.inv(poleplaced_interp.E) @ poleplaced_interp.A))

bode_plot = ld.BodePlot()
bode_plot.add_system(system, color='b')
bode_plot.add_system(poleplaced_interp, color='r', linestyle='dashed')

w_start = -5 # power of 10 to start evaluating frequencies
w_end = 5 # power of 10 to end evaluating frequencies
w_num_points = 10000 # total number of points to evaluate

bode_plot.add_data_tick(at_frequency=10**w_start, pin_to_system_num=0, pin_to_output_num=1, pin_to_input_num=1, markersize=12)
bode_plot.add_data_tick(at_frequency=1, pin_to_system_num=0, pin_to_output_num=1, pin_to_input_num=1, markersize=12)
bode_plot.add_data_tick(at_frequency=100, pin_to_system_num=0, pin_to_output_num=1, pin_to_input_num=1, markersize=12)


bode_plot.show(w_start, w_end, w_num_points)


