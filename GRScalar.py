# GR_Scalar.py
# 
# Library to solve for Einstein's field equations along with a quantum scalar
# field with a trace anomaly.
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


import sys
import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import splrep, splev
import h5py

class System:
    def __init__(self, U=None, V=None, 
        F_tck=None, A_tck=None,
        alpha=0, data_file=None):

        if data_file==None:

            if U==None or V==None:
                print "Provide grid information or load data file."
                sys.exit(1)

            self.alpha = alpha

            self.F_tck = F_tck
            self.A_tck = A_tck

            if len(U.shape) < 2 or len(V.shape) < 2:
                V, U = np.meshgrid(V, U)

            self.U = U; self.V = V
            self.u_i = U[0, 0]; self.u_f = U[-1, 0]
            self.v_i = V[0, 0]; self.v_f = V[0, -1]
            self.Nu = U.shape[0]; self.Nv = V.shape[1]

            self.r = np.zeros([self.Nu, self.Nv])
            self.sigma = np.zeros([self.Nu, self.Nv])
            self.f = np.zeros([self.Nu, self.Nv])
            self.g = np.zeros([self.Nu, self.Nv])
            self.h = np.zeros([self.Nu, self.Nv])
            self.I = np.zeros([self.Nu, self.Nv])
            self.J = np.zeros([self.Nu, self.Nv])
            self.k = np.zeros([self.Nu, self.Nv])

        else:

            saved_data = h5py.File(data_file, "r")

            self.U = saved_data['U'][:]; self.V = saved_data['V'][:]
            self.u_i = self.U[0, 0]; self.u_f = self.U[-1, 0]
            self.v_i = self.V[0, 0]; self.v_f = self.V[0, -1]
            self.Nu = self.U.shape[0]; self.Nv = self.V.shape[1]

            self.r = saved_data['r'][:]
            self.sigma = saved_data['sigma'][:]
            self.f = saved_data['f'][:]
            self.g = saved_data['g'][:]
            self.h = saved_data['h'][:]
            self.I = saved_data['I'][:]
            self.J = saved_data['J'][:]
            self.k = saved_data['k'][:]

            F_array = saved_data['F_array'][:]
            A_array = saved_data['A_array'][:]
            self.F_tck = splrep(self.V[0, :], F_array)
            self.A_tck = splrep(self.U[:, 0], A_array)

            self.alpha = saved_data['alpha'].value

            print "Loaded data file."


    def dX_du(self, y, u, other_variables, solver):

        alpha = self.alpha
        r = y[0]
        sigma = y[1]
        f = y[2]
        g = y[3]
        h = y[4]
        I = y[5]
        k = y[6]

        v = other_variables[0]
        J = other_variables[1]
        F = splev(v, self.F_tck)
        d2A_du2 = splev(u, self.A_tck, 2)
        dA_du = splev(u, self.A_tck, 1)

        dr_du = f

        dsigma_du = I

        df_du = 2*f*I - alpha/r*(J - I**2. + d2A_du2 + dA_du**2.)

        dg_du = -(r/(r**2.- alpha))*(f*g + exp(2*sigma)/4.)

        dh_du = (1./(r**2.-alpha))*(f*g + np.exp(2*sigma)/4.) 

        dI_du = J

        df_dv = dg_du

        dg_dv = 2*g*h - F/r - alpha/r*(k - h**2.)

        dk_du =   (-2.*g/(r**2.-alpha)**2.)*(f*g + np.exp(2*sigma)/4.) 
                + (1./(r**2.-alpha))*(df_dv*g + f*dg_dv + np.exp(2*sigma)*h/2.)

        return [dr_du, dsigma_du, df_du, dg_du, dh_du, dI_du, dk_du]

    def dX_dv(self, y, v, other_variables, classical_source, solver):

        alpha = self.alpha
        r = y[0]
        sigma = y[1]
        f = y[2]
        g = y[3]
        h = y[4]
        I = y[5]
        J = y[6]

        u = other_variables[0]
        k = other_variables[1]
        F = splev(v, self.F_tck)
        d2A_du2 = splev(u, self.A_tck, 2)
        dA_du = splev(u, self.A_tck, 1)

        df_du = 2*f*I - alpha/r*(J - I**2. + d2A_du2 + dA_du**2.)

        dg_du = -(r/(r**2.- alpha))*(f*g + exp(2*sigma)/4.)

        dr_dv = g

        dsigma_dv = h

        df_dv = dg_du

        dg_dv = 2*g*h - F/r - alpha/r*(k - h**2.)

        dh_dv = k

        dI_dv = (1./(r**2.-alpha))*(f*g + np.exp(2*sigma)/4.) # = dH_du

        dJ_dv =   (-2.*f/(r**2.-alpha)**2.)*(f*g + np.exp(2*sigma)/4.) 
                + (1./(r**2.-alpha))*(df_du*g + f*dg_du + np.exp(2*sigma)*I/2.)
        

        return [dr_dv, dsigma_dv, df_dv, dg_dv, dh_dv, dI_dv, dJ_dv]

    def integrate(self, r_break=0, data_dump_filename=None):
        r = self.r
        sigma = self.sigma
        f = self.f
        g = self.g
        h = self.h
        I = self.I
        J = self.J
        k = self.k

        U = self.U; V = self.V
        Nu = self.Nu; Nv = self.Nv

        dX_du = self.dX_du; dX_dv = self.dX_dv

        for u_coord in xrange(Nu-1):
            for v_coord in xrange(Nv-1):

                print "u = ", U[u_coord, v_coord], "v = ", V[u_coord, v_coord]

                # Integrate along u

                r_init = r[u_coord, v_coord + 1]
                sigma_init = sigma[u_coord, v_coord + 1]
                f_init = f[u_coord, v_coord + 1]
                g_init = g[u_coord, v_coord + 1]
                h_init = r[u_coord, v_coord + 1]
                I_init = sigma[u_coord, v_coord + 1]
                J_init = f[u_coord, v_coord + 1]
                k_init = g[u_coord, v_coord + 1]

                init = [r_init, sigma_init, f_init, g_init, h_init, I_init,
                        k_init]
                other_variables = [v, J_init]

                u_init = U[u_coord, v_coord + 1]
                u_final = U[u_coord + 1, v_coord + 1]

                soln_u, info_dict = odeint(dX_du,
                                           init,
                                           [u_init, u_final],
                                           args=(other_variables,),
                                           full_output=True)

                if info_dict['message']!='Integration successful.':
                    print "-"*100
                    print info_dict
                    print "-"*100
                    print "Failed at (u,v) = ", "(", U[u_coord, v_coord +
                        1], ",", V[u_coord, v_coord + 1], ")"
                    print "Indices = ", "(", u_coord, ",", v_coord + 1, ")"
                    if data_dump_filename!=None:
                        self.save_data(data_dump_filename)
                    else:
                        self.save_data("data_dump.hdf5")
                    sys.exit(1)

                # Integrate along v

                r_init = r[u_coord + 1, v_coord]
                sigma_init = sigma[u_coord + 1, v_coord]
                f_init = f[u_coord + 1, v_coord]
                g_init = g[u_coord + 1, v_coord]
                h_init = h[u_coord + 1, v_coord]
                I_init = I[u_coord + 1, v_coord]
                J_init = J[u_coord + 1, v_coord]
                k_init = k[u_coord + 1, v_coord]

                init = [r_init, sigma_init, f_init, g_init, h_init, I_init,
                        J_init]
                other_variables = [u, k_init]

                v_init = V[u_coord + 1, v_coord]
                v_final = V[u_coord + 1, v_coord + 1]

                soln_v, info_dict = odeint(dX_dv,
                                  init,
                                  [v_init, v_final],
                                  args=(other_variables,),
                                  full_output=True)

                if info_dict['message']!='Integration successful.':
                    print "-"*100
                    print info_dict
                    print "-"*100
                    print "Failed at (u,v) = ", "(", U[u_coord + 1, \
                        v_coord], ",", V[u_coord + 1, v_coord], ")"
                    print "Indices = ", "(", u_coord + 1, ",", v_coord, ")"
                    if data_dump_filename!=None:
                        self.save_data(data_dump_filename)
                    else:
                        self.save_data("data_dump.hdf5")
                    sys.exit(1)

                r[u_coord + 1, v_coord + 1] = 0.5*(soln_u[1][0] + soln_v[1][0])
                sigma[u_coord + 1, v_coord + 1] = 
                                            0.5*(soln_u[1][1] + soln_v[1][1])
                f[u_coord + 1, v_coord + 1] = 0.5*(soln_u[1][2] + soln_v[1][2])
                g[u_coord + 1, v_coord + 1] = 0.5*(soln_u[1][3] + soln_v[1][3])
                h[u_coord + 1, v_coord + 1] = 0.5*(soln_u[1][4] + soln_v[1][4])
                I[u_coord + 1, v_coord + 1] = 0.5*(soln_u[1][5] + soln_v[1][5])
                J[u_coord + 1, v_coord + 1] = soln_v[1][6]
                k[u_coord + 1, v_coord + 1] = soln_u[1][6]

                if r[u_coord+1, v_coord+1] < r_break:
                    print "-"*100
                    print "Breaking v loop"
                    print "-"*100
                    break

    def save_data(self, filename):
        U = self.U; V = self.V
        F_array = splev(V[0, :], F_tck)
        A_array = splev(U[:, 0], A_tck)
        alpha = self.alpha
        r = self.r
        sigma = self.sigma
        f = self.f
        g = self.g
        h = self.h
        I = self.I
        J = self.J
        k = self.k

        variables = [U, V, F_array, A_array, alpha,
                     r, sigma, f, g, h, I, J, k]
        variable_names = ['U', 'V', 'F_array', 'A_array',
                          'alpha', 'r', 'sigma', 'f', 'g', 'h', 'I', 'J', 'k']

        datafile = h5py.File(filename, 'w')
        for var, i in enumerate(variable_names):
            datafile[i] = variables[var]
        datafile.close()
