from __future__ import division
import sys
import numpy as np
import h5py
from scipy.interpolate import splev, splrep
from scipy.integrate import odeint

#--------------------ALL PARAMETERS DEFINED HERE-------------------------------#
# Mass
M0 = 10.
v_c = 15.
delta = 0.1

def Mass(V):
    return M0/2 * (1 + np.tanh((V - v_c)/delta))


# L_eps parameters
alpha = 0.1
eps = 0.1
beta = 1.
b = 1.

# Grid parameters
v_i = 0; v_f = 20.
dv = 0.5/16.
#-----------------------END OF PARAMETERS--------------------------------------#
# Grid generation
# Refine in V near the classical shell
v1 = np.arange(v_i, v_c - 4*delta, dv)
v2 = np.arange(v_c - 4*delta, v_c + 4*delta - dv/100., dv/100.)
v3 = np.arange(v_c + 4*delta, v_f, dv)
V = np.concatenate((v1, v2, v3))
#V = np.arange(v_i, v_f, dv)

# Refine in U near the expected classical horizon
horizon = v_c - 4*M0
#U1 = np.arange(-100, -26, 0.1)
#U2 = np.arange(-26, -24, 0.0005)
#U3 = np.arange(-24, -20, 0.1)
#U = np.concatenate([U1, U2, U3])
U = np.arange(-100, -24, 0.05/16.)
u = U

#---------------Generation of initial data with backreaction-------------------#

def L_eps(U):
    M0 = 10.
    return beta/2. * np.log(1/eps**2 * np.tanh(eps*4*M0/(v_c - U - 4*M0))**2 + b**2)


A_tck = splrep(u, L_eps(u))

def A(u, der=0):
    return splev(u, A_tck, der)

A = np.vectorize(A)

dM_du_array = []
u_array = []
def dX_dU(y, u, sigma_tck=None):
    r = y[0]
    f = y[1]
    g = y[2]
    h = y[3]
    k = y[4]

    # Put sigma to a constant, equal to the value at (u_f, v_i). Hence it's
    # derivatives with respect to U are zero. Now since w = 2*dsigma_dU + dA_dU
    # and sigma is a constant, we have w = dA_dU.

    dA_du = A(u, 1)
    d2A_du2 = A(u, 2)

    sigma = 0.
    I = 0.
    J = 0.

    dr_du = f

    df_du = 2*f*I - alpha/r*(J - I**2. + d2A_du2 + dA_du**2.)

    dg_du = -(r/(r**2.- alpha))*(f*g + np.exp(2*sigma)/4.)

    dh_du = (1./(r**2.-alpha))*(f*g + np.exp(2*sigma)/4.) 

    df_dv = dg_du
    dg_dv = 2*g*h - alpha/r*(k - h**2.)
    dk_du =   (-2.*g/(r**2.-alpha)**2.)*(f*g + np.exp(2*sigma)/4.) + (1./(r**2.-alpha))*(df_dv*g + f*dg_dv + np.exp(2*sigma)*h/2.)

    return [dr_du, df_du, dg_du, dh_du, dk_du]

r0 = (v_i - U[0])/2.
f0 = -0.5
g0 = 0.5
h0 = 0.
k0 = 0.
y0 = [r0, f0, g0, h0, k0]
soln = odeint(dX_dU, y0, u, hmax=0.1, mxstep=5000)
r_U_axis = soln[:, 0]; f_U_axis = soln[:, 1]
g_U_axis = soln[:, 2]; h_U_axis = soln[:, 3]
k_U_axis = soln[:, 4]
sigma_U_axis = np.zeros(u.size)
I_U_axis = np.zeros(u.size)
J_U_axis = np.zeros(u.size)

# Setting first a choice for sigma(u0, v)

# Mass function
quantum_mass = r_U_axis/2.*(1 + 4*np.exp(-2*sigma_U_axis)*f_U_axis*g_U_axis)
classical_mass = Mass(V)
total_mass = classical_mass + quantum_mass[0]
total_mass_tck = splrep(V, total_mass)

def dX_dV(y, V):
    r = y[0]
    sigma = y[1]

    mass = splev(V, total_mass_tck)

    dr_dV = 0.5 * (1 - 2*mass/r)
    dsigma_dV = mass/(2 * r**2)

    return [dr_dV, dsigma_dV]

r0 = r_U_axis[0]; sigma0 = sigma_U_axis[0]
y0 = [r0, sigma0]
soln = odeint(dX_dV, y0, V, hmax=0.1, mxstep=5000)
sigma_V_axis = soln[:, 1]
sigma_V_axis_tck = splrep(V, sigma_V_axis)

# Now integrate everything else using the above sigma

def dX_dV(y, V):
    r = y[0]
    f = y[1]
    g = y[2]
    I = y[3]
    J = y[4]

    sigma = splev(V, sigma_V_axis_tck)
    h = splev(V, sigma_V_axis_tck, 1)
    k = splev(V, sigma_V_axis_tck, 2)

    F = splev(V, total_mass_tck, der=1)

    dA_du = A(U[0], 1)
    d2A_du2 = A(U[0], 2)

    df_du = 2*f*I - alpha/r*(J - I**2. + d2A_du2 + dA_du**2.)

    dr_dv = g

    dg_du = -(r/(r**2.- alpha))*(f*g + np.exp(2*sigma)/4.)
    df_dv = dg_du

    dg_dv = 2*g*h - F/r - alpha/r*(k - h**2.)

    dI_dv = (1./(r**2.-alpha))*(f*g + np.exp(2*sigma)/4.) # = dH_du

    dJ_dv =   (-2.*f/(r**2.-alpha)**2.)*(f*g + np.exp(2*sigma)/4.) + (1./(r**2.-alpha))*(df_du*g + f*dg_du + np.exp(2*sigma)*I/2.)

    return [dr_dv, df_dv, dg_dv, dI_dv, dJ_dv]

r0 = r_U_axis[0]
f0 = f_U_axis[0]
g0 = g_U_axis[0]
I0 = I_U_axis[0]
J0 = J_U_axis[0]
y0 = [r0, f0, g0, I0, J0]
soln = odeint(dX_dV, y0, V, hmax=0.1, mxstep=5000)
r_V_axis = soln[:, 0]
f_V_axis = soln[:, 1]
g_V_axis = soln[:, 2]
I_V_axis = soln[:, 3]
J_V_axis = soln[:, 4]
sigma_V_axis = splev(V, sigma_V_axis_tck)
h_V_axis = splev(V, sigma_V_axis_tck, 1)
k_V_axis = splev(V, sigma_V_axis_tck, 2)

datafile = h5py.File('initial_data.hdf5', 'w')
datafile['alpha'] = np.array([alpha])
datafile['U'] = U
datafile['u'] = u
datafile['V'] = V
datafile['u_i'] = U[0]
datafile['v_i'] = v_i

datafile['r_V_axis'] = r_V_axis
datafile['f_V_axis'] = f_V_axis
datafile['g_V_axis'] = g_V_axis
datafile['sigma_V_axis'] = sigma_V_axis
datafile['h_V_axis'] = h_V_axis
datafile['k_V_axis'] = k_V_axis
datafile['I_V_axis'] = I_V_axis
datafile['J_V_axis'] = J_V_axis

datafile['r_U_axis'] = r_U_axis
datafile['f_U_axis'] = f_U_axis
datafile['g_U_axis'] = g_U_axis
datafile['sigma_U_axis'] = sigma_U_axis
datafile['h_U_axis'] = h_U_axis
datafile['k_U_axis'] = k_U_axis
datafile['I_U_axis'] = I_U_axis
datafile['J_U_axis'] = J_U_axis

datafile['classical_mass_V_axis'] = Mass(V)
datafile.close()
