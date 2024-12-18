import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np

import loewner_framework as lf
import loewner_framework.linear_daes as ld


n = 20
m, p = 1, 1
E = np.eye(n)
A = np.random.sample((n,n))
B = np.random.sample((n,m))
C = np.random.sample((p,n))
D = np.random.sample((p,m))
system = ld.LinearDAE(A, B, C, D, label="System 1")
system.label = f"Random ODE (n = {system.n_r})"
#print(system)





Lambda = np.array([ [0, 0, 0],
                    [0, 0, 1],
                    [0, -1, 0]])
R = np.array([[1, 1, 0]])
#W = np.array([[1,2]])
rtd = lf.RightTangentialData(Lambda, R, system)
#print(rtd)

M = np.array([  [0, -100],
                [100, 0]])
L = np.array([  [0],
                [1]])
#V = np.array([[-2],[-1]])
ltd = lf.LeftTangentialData(M, L, system)
#print(ltd)

loewnertd = lf.LoewnerTangentialData(rtd, ltd)
#print(loewnertd)

interpolant_builder = lf.InterpolantFactory(loewnertd)
#print(interpolant_builder)

#interp1 = interpolant_builder.minimal_order(label="Interpolant 1")
#interp1.label += f" (n = {interp1.n_c})"
#print(interp1)

#interp2 = interpolant_builder.minimal_order(D=system.D, label="Interpolant 2")
#interp2.label += f" (n = {interp2.n_c})"
#print(interp2)


possible, shape_dict = interpolant_builder.parameter_dimensions(max(ltd.nu, rtd.rho))
param_dict = {  "D": 2 * (np.random.random_sample(shape_dict["D"]) - 0.5) if shape_dict["D"] else None,
                "P": 2 * (np.random.random_sample(shape_dict["P"]) - 0.5) if shape_dict["P"] else None,
                "Q": 2 * (np.random.random_sample(shape_dict["Q"]) - 0.5) if shape_dict["Q"] else None,
                "G": 2 * (np.random.random_sample(shape_dict["G"]) - 0.5) if shape_dict["G"] else None,
                "T": 2 * (np.random.random_sample(shape_dict["T"]) - 0.5) if shape_dict["T"] else None,
                "H": 2 * (np.random.random_sample(shape_dict["H"]) - 0.5) if shape_dict["H"] else None,
                "F": 2 * (np.random.random_sample(shape_dict["F"]) - 0.5) if shape_dict["F"] else None}
print(shape_dict)
print(param_dict)
possible, total_dim = interpolant_builder.check_consistent_shapes(**param_dict)
print(possible, total_dim)
extended_interp_1 = interpolant_builder.parameterization(**param_dict, label="Interpolant 3")
extended_interp_1.label += f" (n = {extended_interp_1.n_c})"


desired_dimension = max(interpolant_builder.nu, interpolant_builder.rho) + 1
dimension_possible, shapes_dict = interpolant_builder.parameter_dimensions(desired_dimension)
if dimension_possible:
    params_dict = {  "D": 2 * (np.random.random_sample(shapes_dict["D"]) - 0.5) if shapes_dict["D"] else None,
                    "P": 2 * (np.random.random_sample(shapes_dict["P"]) - 0.5) if shapes_dict["P"] else None,
                    "Q": 2 * (np.random.random_sample(shapes_dict["Q"]) - 0.5) if shapes_dict["Q"] else None,
                    "G": 2 * (np.random.random_sample(shapes_dict["G"]) - 0.5) if shapes_dict["G"] else None,
                    "T": 2 * (np.random.random_sample(shapes_dict["T"]) - 0.5) if shapes_dict["T"] else None,
                    "H": 2 * (np.random.random_sample(shapes_dict["H"]) - 0.5) if shapes_dict["H"] else None,
                    "F": 2 * (np.random.random_sample(shapes_dict["F"]) - 0.5) if shapes_dict["F"] else None}
parameters_consistent, associated_dimension = interpolant_builder.check_consistent_shapes(**params_dict)
extended_interp_2 = interpolant_builder.parameterization(**params_dict, label="Interpolant")
extended_interp_2.label += f" (n = {extended_interp_2.n_c})"



bode_plot = ld.BodePlot()
bode_plot.add_system(system, color='b')
bode_plot.add_system(extended_interp_1)#, linestyle='dashed')
bode_plot.add_system(extended_interp_2)#, linestyle='dashdot')

w_start = -4 # power of 10 to start evaluating frequencies
w_end = 3 # power of 10 to end evaluating frequencies
w_num_points = 10000 # total number of points to evaluate

bode_plot.add_data_tick(at_frequency=10**w_start, pin_to_system_num=0, pin_to_output_num=1, pin_to_input_num=1, markersize=12)
bode_plot.add_data_tick(at_frequency=1, pin_to_system_num=0, pin_to_output_num=1, pin_to_input_num=1, markersize=12)
bode_plot.add_data_tick(at_frequency=100, pin_to_system_num=2, pin_to_output_num=1, pin_to_input_num=1, markersize=12)


bode_plot.show(w_start, w_end, w_num_points)


