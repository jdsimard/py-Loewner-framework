import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np

import loewner_framework as lf
import loewner_framework.linear_daes as ld


n = 5
m, p = 3, 2
E = np.eye(n)
A = np.random.sample((n,n))
B = np.random.sample((n,m))
C = np.random.sample((p,n))
D = np.random.sample((p,m))
system = ld.LinearDAE(A, B, C, D, label="System 1")
system.label = f"Random ODE (n = {system.n_r})"
#print(system)

Lambda = np.array([[0, 1],[-1,0]])
R = np.array([[1,0],[0,1],[0,0]])
#W = np.array([[1,2]])
rtd = lf.RightTangentialData(Lambda, R, system)
#print(rtd)

M = np.array([[0, -100],[100, 0]])
L = np.array([[0,1],[1,0]])
#V = np.array([[-2],[-1]])
ltd = lf.LeftTangentialData(M, L, system)
#print(ltd)

loewnertd = lf.LoewnerTangentialData(rtd, ltd)
#print(loewnertd)

interpolant_builder = lf.InterpolantFactory(loewnertd)
#print(interpolant_builder)

interp1 = interpolant_builder.minimal_order(label="Interpolant 1")
#print(interp1)

interp2 = interpolant_builder.minimal_order(D=system.D, label="Interpolant 2")
#print(interp2)








bode_plot = ld.BodePlot()
bode_plot.add_system(system, color='b')
bode_plot.add_system(interp1, color='r', linestyle='dashed')
bode_plot.add_system(interp2, color='g', linestyle='dashdot')

bode_plot.add_data_tick(at_frequency=1, pin_to_system_num=0, pin_to_output_num=1, pin_to_input_num=1, markersize=12)
bode_plot.add_data_tick(at_frequency=100, pin_to_system_num=2, pin_to_output_num=1, pin_to_input_num=1, markersize=12)

w_start = -2 # power of 10 to start evaluating frequencies
w_end = 5 # power of 10 to end evaluating frequencies
w_num_points = 10000 # total number of points to evaluate
bode_plot.show(w_start, w_end, w_num_points)


