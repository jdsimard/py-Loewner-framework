import numpy as np
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import loewner_framework as lf
import loewner_framework.linear_daes as ld



Lambda = complex(imag=1.0) * np.diag([0.1, 10.0, 1000.0])
R = np.array([[1, 1, 1]])
W = np.array([[2, 1, 0.25]])
rtd = lf.RightTangentialData(Lambda, R, W)


M = complex(imag=1.0) * np.diag([0.0, 1.0, 100.0])
L = np.array([  [1],
                [1],
                [1]])
V = np.array([[0.1],[2],[0.5]])
ltd = lf.LeftTangentialData(M, L, V)


loewnertd = lf.LoewnerTangentialData(rtd, ltd)
interpolant_builder = lf.InterpolantFactory(loewnertd)


possible, shape_dict = interpolant_builder.parameter_dimensions(max(rtd.rho, ltd.nu))
param_dict = {  "D": 2 * (np.random.random_sample(shape_dict["D"]) - 0.5) if shape_dict["D"] else None,
                "P": 2 * (np.random.random_sample(shape_dict["P"]) - 0.5) if shape_dict["P"] else None,
                "Q": 2 * (np.random.random_sample(shape_dict["Q"]) - 0.5) if shape_dict["Q"] else None,
                "G": 2 * (np.random.random_sample(shape_dict["G"]) - 0.5) if shape_dict["G"] else None,
                "T": 2 * (np.random.random_sample(shape_dict["T"]) - 0.5) if shape_dict["T"] else None,
                "H": 2 * (np.random.random_sample(shape_dict["H"]) - 0.5) if shape_dict["H"] else None,
                "F": 2 * (np.random.random_sample(shape_dict["F"]) - 0.5) if shape_dict["F"] else None}
param_dict["D"] = None

possible, total_dim = interpolant_builder.check_consistent_shapes(**param_dict)
interp = interpolant_builder.parameterization(**param_dict, label="Interpolant")
interp.label += f" (n = {interp.n_c})"



bode_plot = ld.BodePlot()
bode_plot.add_system(interp, color='k', linestyle='solid')

w_start = -5 # power of 10 to start evaluating frequencies
w_end = 4 # power of 10 to end evaluating frequencies
w_num_points = 100000 # total number of points to evaluate

bode_plot.add_data_tick(at_frequency=10**w_start, pin_to_system_num=0, pin_to_output_num=1, pin_to_input_num=1, markersize=12, color='r')
bode_plot.add_data_tick(at_frequency=0.1, pin_to_system_num=0, pin_to_output_num=1, pin_to_input_num=1, markersize=12, color='r')
bode_plot.add_data_tick(at_frequency=1, pin_to_system_num=0, pin_to_output_num=1, pin_to_input_num=1, markersize=12, color='r')
bode_plot.add_data_tick(at_frequency=10, pin_to_system_num=0, pin_to_output_num=1, pin_to_input_num=1, markersize=12, color='r')
bode_plot.add_data_tick(at_frequency=100, pin_to_system_num=0, pin_to_output_num=1, pin_to_input_num=1, markersize=12, color='r')
bode_plot.add_data_tick(at_frequency=1000, pin_to_system_num=0, pin_to_output_num=1, pin_to_input_num=1, markersize=12, color='r')

bode_plot.show(w_start, w_end, w_num_points)
