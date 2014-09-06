from __future__ import division
import sys
import numpy as np
import h5py
from scipy.interpolate import splev, splrep
from scipy.integrate import odeint

#--------------------ALL PARAMETERS DEFINED HERE-------------------------------#
# Mass
M0 = 10. + 0.5944099 + 0.095610 + 0.01537 + 0.002459 + 0.00042 + 0.2 - 0.025 + \
0.001 - 0.001# Total mass, eps = 1e-3, alpha = 0.1
#M0 = M0 + 0.6 + 1.63862 - 1. + 0.1 + 0.045 + 0.0015
v_c = 15.
delta = 0.1
#M0 = 10. - 0.2
#M0 = 10.

def Mass(V):
    return M0/2 * (1 + np.tanh((V - v_c)/delta))


# L_eps parameters
#alpha = 0.00001
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

#---------------Generation of initial data without backreaction----------------#
#def dX_dV(y, V):
#    r = y[0]
#    sigma = y[1]
#
#    dr_dV = 0.5 * (1 - 2*Mass(V)/r)
#    dsigma_dV = Mass(V)/(2 * r**2)
#
#    return [dr_dV, dsigma_dV]
#
#r0 = (V[0] - u_i)/2.; sigma0 = 0.; y0 = [r0, sigma0]
#soln = odeint(dX_dV, y0, V, hmax=0.1, mxstep=5000)
#r = soln[:, 0]; sigma = soln[:, 1]
#r_tck = splrep(V, r); sigma_tck = splrep(V, sigma)
#g = splev(V, r_tck, der=1)
#d = splev(V, sigma_tck, der=1)
#f = -0.5 * np.exp(2*sigma)
#
#datafile = h5py.File('initial_data.hdf5', 'w')
#datafile['u_i'] = u_i
#datafile['U'] = U
#datafile['V'] = V
#datafile['r_V_axis'] = r
#datafile['f_V_axis'] = f
#datafile['g_V_axis'] = g
#datafile['sigma_V_axis'] = sigma
#datafile['d_V_axis'] = d
#datafile['classical_mass_V_axis'] = Mass(V)
#datafile['w_V_axis'] = np.zeros(V.shape)
#datafile['z_V_axis'] = np.zeros(V.shape)
#datafile['r_U_axis'] = (V[0] - U)/2.
#datafile['f_U_axis'] = -0.5 * np.ones(U.shape)
#datafile['g_U_axis'] = 0.5 * np.ones(U.shape)
#datafile['sigma_U_axis'] = np.zeros(U.shape)
#datafile['d_U_axis'] = np.zeros(U.shape)
#datafile['w_U_axis'] = np.zeros(U.shape)
#datafile['z_U_axis'] = np.zeros(U.shape)

# ---------------- Alternative gauge ----------------
#def dX_dv(y, v):
#    r = y[0]
#    g = y[1]
#    f = y[2]
#    w = y[3]
#
#    dr_dv = g
#    dg_dv = -source(v)/r
#    df_dv = -(f*g + 1/4.)/(r*(1 - alpha/r**2))
#    dw_dv = 2*(f*g + 1/4.)/(r**2 - alpha)
#    return [dr_dv, dg_dv, df_dv, dw_dv]
#
#ri = (v_i - u_i)/2.; gi = 0.5; fi = -0.5; wi = 0; y0 = [ri, gi, fi, wi]
#soln = odeint(dX_dv, y0, v)
#r = soln[:, 0]; g = soln[:, 1]; f = soln[:, 2]; w = soln[:, 3]
#system.r[0, :] = r; system.g[0, :] = g; system.f[0, :] = f; system.w[0, :] = w
#system.sigma[0, :] = 0; system.d[0, :] = 0
# ---------------- End of alternative gauge -----------


# ---------------- Code to read initial data generated by integrating back
# solution of scalar field in region II [Alternative scheme] -----------------
#scalar_field_initial_data = h5py.File('test_initial_data_eps_0.0001.hdf5', 'r')
#w_u_axis = scalar_field_initial_data['initial_data_u_axis'][:]
#w_v_axis = scalar_field_initial_data['initial_data_v_axis'][:]
#
#z_v_axis = np.concatenate((w_u_axis, np.zeros(system.Nv))) #After reflection
#
#w_u_axis_tck = splrep(system.U[:, 0], w_u_axis)
#
#z_v_axis_tck = w_u_axis_tck
#
#source_tck = splrep(system.V[0, :], source(system.V[0, :]))
#
#def dX_dv(y, v):
#    f = y[0]
#    sigma = y[1]
#    w = y[2]
#
#    print "v = ", v
#
#    g = 0.5
#    r = (v - u_i)/2
#    z = splev(v, z_v_axis_tck)
#    dz_dv = splev(v, z_v_axis_tck, der=1)
#
#    d = (alpha*(z**2/4. + dz_dv/2.))/(r + alpha*z)
#
#    df_dv = -r*(f*g + np.exp(2*sigma)/4.)/(r**2 - alpha)
#    dsigma_dv = d
#    dw_dv = 2*(f*g + np.exp(2*sigma)/4.)/(r**2 - alpha)
#
#    return [df_dv, dsigma_dv, dw_dv]
#
#fi = -0.5; sigmai = 0.0; wi = 0.0; y0 = [fi, sigmai, wi]
#soln = odeint(dX_dv, y0, u[10:], hmax=0.1, mxstep=5000)
#
#
#system.r[0, :] = r; system.f[0, :] = f; system.g[0, :] = 0.5; system.d[0, :] = d
#system.sigma[0, :] = sigma; system.z[0, :] = z; system.w[0, :] = w
#
#def dX_du(y, u):
#    r = y[0]
#    f = y[1]
#    d = y[2]
#    z = y[3]
#
#    print "u = ", u
#
#    w_u_axis = splev(u, w_u_axis_tck)
#    dw_du_u_axis = splev(u, w_u_axis_tck, der=1)
#
#    dr_du = f
#    df_du = -alpha*(0.25*w_u_axis**2 + 0.5*dw_du_u_axis)/r
#    dd_du = (f*0.5 + 0.25)/(r**2 - alpha)
#    dz_du = 2*(f*0.5 + 0.25)/(r**2 - alpha)
#
#    return [dr_du, df_du, dd_du, dz_du]
#
#ri = system.r[0, 0]; fi = system.f[0, 0]; di = system.d[0, 0]
#zi = system.z[0, 0]; y0 = [ri, fi, di, zi]
#soln = odeint(dX_du, y0, u, hmax=0.1, mxstep=5000)
#r = soln[:, 0]; f = soln[:, 1]; d = soln[:, 2]; z = soln[:, 3]
#system.r[:, 0] = r; system.f[:, 0] = f; system.d[:, 0] = d; system.z[:, 0] = z
#system.g[:, 0] = 0.5; system.sigma[:, 0] = 0.
#system.w[:, 0] = w_u_axis
#
# ------------End of code to read initial data [Alternative scheme]-----------

# ---------------- Code to read initial data generated by integrating back
# solution of scalar field in region II ----------------------------------
#scalar_field_initial_data = h5py.File('test_initial_data_eps_0.001.hdf5', 'r')
#w_u_axis = scalar_field_initial_data['initial_data_u_axis'][:]
#w_v_axis = scalar_field_initial_data['initial_data_v_axis'][:]
#
#w_u_axis_tck = splrep(system.U[:, 0], w_u_axis)
#
#def dX_du(y, u):
#    r = y[0]
#    f = y[1]
#    d = y[2]
#    z = y[3]
#
#    print "u = ", u
#
#    w_u_axis = splev(u, w_u_axis_tck)
#    dw_du_u_axis = splev(u, w_u_axis_tck, der=1)
#
#    dr_du = f
#    df_du = -alpha*(0.25*w_u_axis**2 + 0.5*dw_du_u_axis)/r
#    dd_du = (f*0.5 + 0.25)/(r**2 - alpha)
#    dz_du = 2*(f*0.5 + 0.25)/(r**2 - alpha)
#
#    return [dr_du, df_du, dd_du, dz_du]
#
#ri = (v_i - u_i)/2.; fi = -0.5; di = 0.; zi = 0.; y0 = [ri, fi, di, zi]
#soln = odeint(dX_du, y0, u, hmax=0.1, mxstep=5000)
#r = soln[:, 0]; f = soln[:, 1]; d = soln[:, 2]; z = soln[:, 3]
#system.r[:, 0] = r; system.f[:, 0] = f; system.d[:, 0] = d; system.z[:, 0] = z
#system.g[:, 0] = 0.5; system.sigma[:, 0] = 0
#system.w[:, 0] = w_u_axis; system.w[0, :] = w_v_axis[::-1]

#def dX_du(y, u):
#    print "u = ", u
#
#    w = splev(u, w_u_axis_tck)
#    dw_du = splev(u, w_u_axis_tck, der=1)
#    r = -u/2.
#
#    dsigma_du = -alpha*(w**2/4 + dw_du/2.)/(r - alpha*w)
#    print "r - alpha*w = ", r - alpha*w
#
#    return [dsigma_du]
#
#
#soln = odeint(dX_du, [0], u, hmax=0.1, mxstep=5000)
#
# -----------------End of code to read initial data------------------------

# -----------------Code to read analytical scalar field data on the boundary---
# Parameters for scalar field boundary condition
#a_0_s = 1
#u_c_s = -42.8
#delta_s = 1/10.
#def phi_u_axis(U): return a_0_s * np.exp(-(U - u_c_s)**2.0/delta_s**2.0)
#
#phi_u_axis_tck = splrep(u, phi_u_axis(u))
#
#def dX_du(y, u):
#    r = y[0]
#    f = y[1]
#    d = y[2]
#    z = y[3]
#
#    print "u = ", u
#
#    w_u_axis = splev(u, phi_u_axis_tck, der=1)
#    dw_du_u_axis = splev(u, phi_u_axis_tck, der=2)
#
#    dr_du = f
#    df_du = -alpha*(0.25*w_u_axis**2 + 0.5*dw_du_u_axis)/r
#    dd_du = (f*0.5 + 0.25)/(r**2 - alpha)
#    dz_du = 2*(f*0.5 + 0.25)/(r**2 - alpha)
#
#    return [dr_du, df_du, dd_du, dz_du]
#
#ri = (v_i - u_i)/2.; fi = -0.5; di = 0.; zi = 0.; y0 = [ri, fi, di, zi]
#soln = odeint(dX_du, y0, u, hmax=0.1, mxstep=5000)
#r = soln[:, 0]; f = soln[:, 1]; d = soln[:, 2]; z = soln[:, 3]
#system.r[:, 0] = r; system.f[:, 0] = f; system.d[:, 0] = d; system.z[:, 0] = z
#system.g[:, 0] = 0.5; system.sigma[:, 0] = 0; system.w[:, 0] = splev(u,
#        phi_u_axis_tck, der=1)
#
# ----------End of Code----------------------------
