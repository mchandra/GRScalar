from __future__ import division
import sys
import numpy as np
import h5py
from scipy.interpolate import splev, splrep
from scipy.integrate import odeint

#--------------------ALL PARAMETERS DEFINED HERE-------------------------------#
# Mass
#M0 = 10. + 0.5944099 + 0.095610 + 0.01537 + 0.002459 + 0.00042 + 0.2 - 0.025 + \
#0.001 - 0.001# Total mass, eps = 1e-3, alpha = 0.1
M0 = 10.
v_c = 15.
delta = 0.1

def Mass(V):
    return M0/2 * (1 + np.tanh((V - v_c)/delta))


# L_eps parameters
alpha = 0.1
eps = 0.001
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

# Refine in U near the expected classical horizon
horizon = v_c - 4*M0
U1 = np.arange(-100, -26, 0.1)
U2 = np.arange(-26, -24, 0.0005)
U3 = np.arange(-24, -20, 0.1)
U = np.concatenate([U1, U2, U3])
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
    d = y[2]
    z = y[3]
    g = y[4]
    M = y[5]

    # Put sigma to a constant, equal to the value at (u_f, v_i). Hence it's
    # derivatives with respect to U are zero. Now since w = 2*dsigma_dU + dA_dU
    # and sigma is a constant, we have w = dA_dU.

    dA_du = A(u, 1)
    d2A_du2 = A(u, 2)

    if sigma_tck==None:
        sigma = 0.
        dsigma_du = 0.
        d2sigma_du2 = 0.
    else:
        sigma = splev(u, sigma_tck)
        dsigma_du = splev(u, sigma_tck, der=1)
        d2sigma_du2 = splev(u, sigma_tck, der=2)

    dsigma_dU = dsigma_du
    d2sigma_dU2 = d2sigma_du2

    w = 2*dsigma_dU + dA_du
    dw_dU = 2*d2sigma_dU2 + d2A_du2

    T_UU = alpha*(w**2/4. + dw_dU/2. - w*dsigma_dU)
    T_UV =  -alpha*(f*g + np.exp(2*sigma)/4.)/(r**2 - alpha)


    dr_du = f
    df_du = 2*f*dsigma_dU -alpha*(0.25*w**2 + 0.5*dw_dU)/r
    dd_du = (f*g + np.exp(2*sigma)/4.)/(r**2*(1 - alpha/r**2))
    dz_du = 2*(f*g + np.exp(2*sigma)/4.)/(r**2 - alpha)
    dg_du = -(f*g + np.exp(2*sigma)/4.)/(r*(1 - alpha/r**2))
    dM_du = (2*np.exp(-2*sigma)*f*T_UV - 2*np.exp(-2*sigma)*g*T_UU)
    if sigma_tck==None:
        dM_du_array.append(dM_du)
        u_array.append(u)

    return [dr_du, df_du, dd_du, dz_du, dg_du, dM_du]

r0 = (v_i - U[-1])/2.; f0 = -0.5
d0 = 0; z0 = 0.
g0 = 0.5; Minit = 0.
y0 = [r0, f0, d0, z0, g0, Minit]
soln = odeint(dX_dU, y0, u[::-1], hmax=0.1, mxstep=5000)
M_U_axis = soln[::-1, 5]
r_old = soln[::-1, 0]; f_old = soln[::-1, 1]
d_old = soln[::-1, 2]; z_old = soln[::-1, 3]
g_old = soln[::-1, 4]; M_old = soln[::-1, 5]

M_U_axis_tck = splrep(u, M_U_axis)

def dX1_dU(y, u):
    r = y[0]
    sigma = y[1]

    mass = splev(u, M_U_axis_tck)

    dr_du = -0.5 * (1 - 2*mass/r)
    dsigma_du = -mass/(2 * r**2)

    return [dr_du, dsigma_du]

r0 = (v_i - U[-1])/2.; sigma0 = 0.
y0 = [r0, sigma0]
soln = odeint(dX1_dU, y0, u[::-1], hmax=0.1, mxstep=5000)
sigma_U_axis = 0.*soln[::-1, 1]
sigma_U_axis_tck = splrep(u, sigma_U_axis)

y0 = [r0, f0, d0, z0, g0, Minit]
soln = odeint(dX_dU, y0, u[::-1], hmax=0.1, mxstep=5000, args=(sigma_U_axis_tck,))
r_U_axis = soln[::-1, 0]; f_U_axis = soln[::-1, 1]
d_U_axis = soln[::-1, 2]; z_U_axis = soln[::-1, 3]
g_U_axis = soln[::-1, 4]; M_U_axis = soln[::-1, 5]

# Mass function
quantum_mass = r_U_axis/2.*(1 + 4*np.exp(-2*sigma_U_axis)*f_U_axis*g_U_axis)
classical_mass = Mass(V)
total_mass = classical_mass + quantum_mass[0]
total_mass_tck = splrep(V, total_mass)

def dX1_dV(y, V):
    r = y[0]
    sigma = y[1]

    mass = splev(V, total_mass_tck)

    dr_dV = 0.5 * (1 - 2*mass/r)
    dsigma_dV = mass/(2 * r**2)

    return [dr_dV, dsigma_dV]

r0 = r_U_axis[0]; sigma0 = sigma_U_axis[0]; y0 = [r0, sigma0]
soln = odeint(dX1_dV, y0, V, hmax=0.1, mxstep=5000)
sigma_V_axis = soln[:, 1]
sigma_V_axis_tck = splrep(V, sigma_V_axis)

def dX2_dV(y, V):
    r = y[0]
    g = y[1]
    f = y[2]
    w = y[3]
    M = y[4]

    dB_dV = 0.
    d2B_dV2 = 0.

    sigma = splev(V, sigma_V_axis_tck)
    d = splev(V, sigma_V_axis_tck, 1)
    dd_dV = splev(V, sigma_V_axis_tck, 2)
    z = 2*d + dB_dV
    dz_dV = 2*dd_dV + d2B_dV2

    F = splev(V, total_mass_tck, der=1)

    T_VV = alpha*(z**2/4. + dz_dV/2. - z*d) + F
    T_UV =  -alpha*(f*g + np.exp(2*sigma)/4.)/(r**2 - alpha)

    dr_dV = g
    dg_dV = 2*d*g - F/r - alpha*(0.25*z**2 + 0.5*dz_dV - z*d)/r
    df_dV = -(f*g + np.exp(2*sigma)/4.)/(r*(1 - alpha/r**2))
    dw_dV = 2*(f*g + np.exp(2*sigma)/4.)/(r**2 - alpha)
    dM_dV = 2*np.exp(-2*sigma)*g*T_UV - 2*np.exp(-2*sigma)*f*T_VV
#    print "V = ", V
#    print "r = ", r
#    print "r - alpha/r = ", r - alpha/r
#    print "r**2 - alpha = ", r**2 - alpha

    return [dr_dV, dg_dV, df_dV, dw_dV, dM_dV]

r0 = r_U_axis[0]; g0 = g_U_axis[0]; f0 = f_U_axis[0]; w0 = 0.; Minit = quantum_mass[0]
y0 = [r0, g0, f0, w0, Minit]
soln = odeint(dX2_dV, y0, V, hmax=0.1, mxstep=5000)
r_V_axis = soln[:, 0]; g_V_axis = soln[:, 1]
f_V_axis = soln[:, 2]; w_V_axis = soln[:, 3]
M_V_axis = soln[:, 4]

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
datafile['sigma_V_axis'] = splev(V, sigma_V_axis_tck)
datafile['d_V_axis'] = splev(V, sigma_V_axis_tck, der=1)
datafile['w_V_axis'] = w_V_axis
datafile['z_V_axis'] = 2*splev(V, sigma_V_axis_tck, der=1)
datafile['classical_mass_V_axis'] = Mass(V)
datafile['r_U_axis'] = r_U_axis
datafile['f_U_axis'] = f_U_axis
datafile['g_U_axis'] = g_U_axis
datafile['sigma_U_axis'] = sigma_U_axis
datafile['d_U_axis'] = d_U_axis
datafile['w_U_axis'] = A(u, 1)
datafile['z_U_axis'] = z_U_axis
datafile['M_U_axis'] = M_U_axis
datafile['M_V_axis'] = M_V_axis
datafile['quantum_mass'] = quantum_mass
datafile['r_old'] = r_old
datafile['f_old'] = f_old
datafile['d_old'] = d_old
datafile['g_old'] = g_old
datafile['M_old'] = M_old
datafile.close()
np.savetxt('dM_du_array.dat', dM_du_array)
np.savetxt('u_array.dat', u_array)
