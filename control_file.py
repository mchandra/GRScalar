# control_file.py
# 
# Script using GR_Scalar library for numerical solutions to Einstein's field
# equations.
# 
# Copyright (C) 2011 Mani Chandra <mc0710@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import GRScalar_test
from scipy.interpolate import splrep, splev
import h5py

# Import parameters and initial data from the initial data file.
initial_data = h5py.File('initial_data.hdf5', 'r')

# Quantum length scale
alpha = initial_data['alpha'][:][0]

# Full scalar field evolution ?
EvolveScalarField = 1

# Grid generation
u = initial_data['u'][:]
V = initial_data['V'][:]

Nu = u.size; Nv = V.size

print "Nu = ", Nu
print "Nv = ", Nv

# Parameters for dust shell

Mass = initial_data['classical_mass_V_axis'][:]
Mass_tck = splrep(V, Mass)

def source(V): return splev(V, Mass_tck, der=1)

# Parameters for scalar field boundary condition
def phi(V): return 0.

# Data file name
#data_file = "collapse_with_backreaction_PP_scheme_eps_0.001_alpha_0.1_V_300_u_750_sigma_zero.hdf5"
#data_file = "classical_collapse_PP_scheme_V_1000_u_750.hdf5"
data_file = \
"test4_with_Leps_eps_0.001_1.hdf5"
#"backreaction_without_L_eps_M0_10.hdf5"
#"backreaction_with_L_eps_sign_changed_test_eps_0.001_alpha_0.2.hdf5"

system = GRScalar_test.System(u, V,
    alpha=alpha, FullScalarFieldEvolution=EvolveScalarField, 
    classical_source=source, scalar_field=phi)

system.r[0, :] = initial_data['r_V_axis'][:]
system.f[0, :] = initial_data['f_V_axis'][:]
system.g[0, :] = initial_data['g_V_axis'][:]
system.sigma[0, :] = initial_data['sigma_V_axis'][:]
system.d[0, :] = initial_data['d_V_axis'][:]
system.w[0, :] = initial_data['w_V_axis'][:]
system.z[0, :] = initial_data['z_V_axis'][:]

system.r[:, 0] = initial_data['r_U_axis'][:]
system.f[:, 0] = initial_data['f_U_axis'][:]
system.g[:, 0] = initial_data['g_U_axis'][:]
system.sigma[:, 0] = initial_data['sigma_U_axis'][:]
system.d[:, 0] = initial_data['d_U_axis'][:]
system.w[:, 0] = initial_data['w_U_axis'][:]
system.z[:, 0] = initial_data['z_U_axis'][:]

system.integrate(data_dump_filename=data_file, r_break=5.)
system.save_data(data_file)
