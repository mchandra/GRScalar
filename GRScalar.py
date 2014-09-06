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
from scipy.interpolate import splrep, splev, interp1d
import h5py

class System:
    def __init__(self, U=None, V=None, 
        classical_source=None, scalar_field=None,
        alpha=0, FullScalarFieldEvolution=0, 
        data_file=None):

        if data_file==None:

            if U==None or V==None:
                print "Provide grid information or load data file."
                sys.exit(1)

            self.alpha = alpha
            self.FullScalarFieldEvolution = FullScalarFieldEvolution

            if classical_source==None:
                self.classical_source = lambda x: 0
            else:
                self.classical_source = classical_source

            if scalar_field==None:
                self.scalar_field = lambda x: 0
            else:
                self.scalar_field = scalar_field

            if len(U.shape) < 2 or len(V.shape) < 2:
                V, U = np.meshgrid(V, U)

            self.U = U; self.V = V
            self.u_i = U[0, 0]; self.u_f = U[-1, 0]
            self.v_i = V[0, 0]; self.v_f = V[0, -1]
            self.Nu = U.shape[0]; self.Nv = V.shape[1]

            # Array initialization
            # r = r(u, v), f = dr/du, g = dr/dv
            # sigma = sigma(u, v), d = dsigma/dv
            # phi = phi(u, v), w = dphi/du, z = dphi/dv

            self.r = np.zeros([self.Nu, self.Nv])
            self.f = np.zeros([self.Nu, self.Nv])
            self.g = np.zeros([self.Nu, self.Nv])
            self.sigma = np.zeros([self.Nu, self.Nv])
            self.d = np.zeros([self.Nu, self.Nv])
            self.w = np.zeros([self.Nu, self.Nv])
            self.z = np.zeros([self.Nu, self.Nv])

#            print "Setting Initial and boundary data..."
#            self.set_boundary_conditions()
#            print "...done."

        else:

            saved_data = h5py.File(data_file, "r")

            self.U = saved_data['U'][:]; self.V = saved_data['V'][:]
            self.u_i = self.U[0, 0]; self.u_f = self.U[-1, 0]
            self.v_i = self.V[0, 0]; self.v_f = self.V[0, -1]
            self.Nu = self.U.shape[0]; self.Nv = self.V.shape[1]

            self.r = saved_data['r'][:]
            self.f = saved_data['f'][:]
            self.g = saved_data['g'][:]
            self.sigma = saved_data['sigma'][:]
            self.d = saved_data['d'][:]
            self.w = saved_data['w'][:]
            self.z = saved_data['z'][:]

            source_array = saved_data['source_array'][:]
            scalar_field_array = saved_data['scalar_field_array'][:]
            self.classical_source = interp1d(self.V[0, :], source_array)
            self.scalar_field = interp1d(self.V[0, :], scalar_field_array)

            self.alpha = saved_data['alpha'].value
            self.FullScalarFieldEvolution = \
                saved_data['FullScalarFieldEvolution'].value

            print "Loaded data file."

    def set_boundary_conditions(self):
        U = self.U; V = self.V
        u_i = self.u_i; v_i = self.v_i

        classical_source = self.classical_source
        phi = self.scalar_field

        alpha = self.alpha
        FullScalarFieldEvolution = self.FullScalarFieldEvolution

        r = self.r
        f = self.f
        g = self.g
        sigma = self.sigma
        d = self.d
        z = self.z
        w = self.w

        r[0, :] = (V[0, :] - u_i)/2.0
        r[:, 0] = (v_i - U[:, 0])/2.0
        g[:, 0] = 1/2.0
        g[0, :] = 1/2.0

        # Boundary conditions for z

        if FullScalarFieldEvolution == 1:

            tck_phi, fp, ierr, msg = splrep(V[0, :], phi(V[0, :]), k=3, 
                full_output=True)

            if ierr > 0:
                print "Error in spline representation used for derivative \
                    calculation"
                sys.exit(1)

            z[0, :] = splev(V[0, :], tck_phi, der=1)

        # Boundary conditions for d
        d[:, 0] = 0.

        if alpha == 0:
            d[0, :] = classical_source(V[0, :])/r[0, :]

        elif FullScalarFieldEvolution == 0:
            def dd_dv_v_axis(y, v):
                r = (v - u_i)/2.
                return r/alpha*(y[0] - classical_source(v)/r) + y[0]**2

            d[0, :] = odeint(dd_dv_v_axis, d[0, 0], V[0, :], hmax=0.1)[:, 0]

            d_v_axis = interp1d(V[0, :], d[0, :])

        elif FullScalarFieldEvolution == 1:
            dz_dv_v_axis = splev(V[0, :], tck_phi, der=2)

            d[0, :] = (classical_source(V[0, :]) + alpha*(z[0, :]**2/4. +
                dz_dv_v_axis/2.))/(r[0, :] + alpha*z[0, :])

            d_v_axis = interp1d(V[0, :], d[0, :])

        # Boundary conditions for sigma
        sigma[:, 0] = 0.

        def dsigma_dv_v_axis(y, v):
            if alpha == 0:
                r = (v - u_i)/2.
                return classical_source(v)/r

            else:
                if v < V[0, -1]:
                    return d_v_axis(v)
                else:
                    return d_v_axis(V[0, -1])

        sigma[0, :] = odeint(dsigma_dv_v_axis, sigma[0, 0], V[0, :], hmax=0.1)[:, 0]

        sigma_v_axis = interp1d(V[0, :], sigma[0, :])

        # Boundary conditions for f
        f[:, 0] = -1/2.

        def df_dv_v_axis(y, v):
            r = (v - u_i)/2.
            if v < V[0, -1]:
                sigma_local = sigma_v_axis(v)
            else:
                sigma_local = sigma_v_axis(V[0, -1])
            return -(y[0]*0.5 + np.exp(2*sigma_local)/4.)/(r*(1 - alpha/r**2))

        f[0, :] = odeint(df_dv_v_axis, f[0, 0], V[0, :], hmax=0.1)[:, 0]

        f_v_axis = interp1d(V[0, :], f[0, :])

        # Boundary conditions for w

        def dw_dv_v_axis(y, v):
            r = (v - u_i)/2.
            if v < V[0, -1]:
                f_local = f_v_axis(v)
                sigma_local = sigma_v_axis(v)
            else:
                f_local = f_v_axis(V[0, -1])
                sigma_local = sigma_v_axis(V[0, -1])
            return 2*(f_local*0.5 + np.exp(2*sigma_local)/4.)/(r**2 - alpha)

        w[0, :] = odeint(dw_dv_v_axis, w[0, 0], V[0, :], hmax=0.1)[:, 0]

    def dX_du(self, y, u, other_variables, solver):

        alpha = self.alpha
        if solver=='lsoda':
            r_local = y[0]

        elif solver=='vode':
            r_local = u[0]

        f_local = other_variables[0]
        g_local = other_variables[1]
        sigma_local = other_variables[2]


        dr_du = f_local
        dd_du = (f_local*g_local/(r_local**2) + \
            np.exp(2*sigma_local)/(4*r_local**2))*(1./(1 - alpha/r_local**2))
        dz_du = 2*(f_local*g_local + np.exp(2*sigma_local)/4.)/(r_local**2 -
            alpha)

        return [dr_du, dd_du, dz_du]

    def dX_dv(self, y, v, other_variables, classical_source, solver):

        alpha = self.alpha
        if solver=='lsoda':
            r_local = y[0]
            f_local = y[1]
            g_local = y[2]
            sigma_local = y[3]
        elif solver=='vode':
            r_local = v[0]
            f_local = v[1]
            g_local = v[2]
            sigma_local = v[3]


        d_local = other_variables[0]
        z_local = other_variables[1]
        F = classical_source

        dr_dv = g_local
        df_dv = (-f_local*g_local/r_local - \
            np.exp(2*sigma_local)/(4*r_local))*(1/(1 - alpha/r_local**2))

        if self.FullScalarFieldEvolution:
            dz_dv_local = other_variables[2]
            dg_dv = 2*d_local*g_local - F/r_local - alpha*(z_local**2/4. +
                dz_dv_local/2. - z_local*d_local)/r_local
        else:
            dd_dv_local = other_variables[2]
            dg_dv = 2*d_local*g_local - F/r_local - alpha*(dd_dv_local -
                d_local**2)/r_local

        dsigma_dv = d_local
        dw_dv = 2*(f_local*g_local + np.exp(2*sigma_local)/4.)/(r_local**2 -
            alpha)

        return [dr_dv, df_dv, dg_dv, dsigma_dv, dw_dv]

    def integrate(self, r_break=0, data_dump_filename=None):
        r = self.r
        d = self.d
        z = self.z

        f = self.f
        g = self.g
        sigma = self.sigma
        w = self.w

        U = self.U; V = self.V
        Nu = self.Nu; Nv = self.Nv

        dX_du = self.dX_du; dX_dv = self.dX_dv

        # Restricting to lsoda for now.
        solver = 'lsoda'

        if solver=='lsoda':

            for u_coord in xrange(Nu-1):
                for v_coord in xrange(Nv-1):

                    print "u = ", U[u_coord, v_coord], "v = ", V[u_coord,
                        v_coord]

                    # Integrate along u

                    r_init = r[u_coord, v_coord + 1]
                    d_init = d[u_coord, v_coord + 1]
                    z_init = z[u_coord, v_coord + 1]

                    f_init = f[u_coord, v_coord + 1]
                    g_init = g[u_coord, v_coord + 1]
                    sigma_init = sigma[u_coord, v_coord + 1]

                    other_variables = [f_init, g_init, sigma_init]

                    u_init = U[u_coord, v_coord + 1]
                    u_final = U[u_coord + 1, v_coord + 1]

                    soln_u, info_dict = odeint(dX_du, [r_init, d_init,
                        z_init], [u_init, u_final], args=(other_variables,
                        solver,), full_output=True)

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

                    r[u_coord + 1, v_coord + 1] = soln_u[1][0]
                    d[u_coord + 1, v_coord + 1] = soln_u[1][1]
                    z[u_coord + 1, v_coord + 1] = soln_u[1][2]

                    # Integrate along v

                    r_init = r[u_coord + 1, v_coord]
                    f_init = f[u_coord + 1, v_coord]
                    g_init = g[u_coord + 1, v_coord]
                    sigma_init = sigma[u_coord + 1, v_coord]
                    w_init = w[u_coord + 1, v_coord]

                    d_init = d[u_coord + 1, v_coord]
                    z_init = z[u_coord + 1, v_coord]

                    if self.FullScalarFieldEvolution:
                        if v_coord==0:
                            dz_dv_init = 0.
                        else:
                            dz_dv_init = self.derivative('z', 'v', [u_coord + 1,
                                v_coord] , order=1)
                        other_variables = [d_init, z_init, dz_dv_init]
                    else:
                        if v_coord==0:
                            dd_dv_init = 0.
                        else:
                            dd_dv_init = self.derivative('d', 'v', [u_coord + 1,
                                v_coord], order=1)
                        other_variables = [d_init, z_init, dd_dv_init]

                    F = self.classical_source(V[u_coord + 1, v_coord])

                    v_init = V[u_coord + 1, v_coord]
                    v_final = V[u_coord + 1, v_coord + 1]

                    soln_v, info_dict = odeint(dX_dv, [r_init, f_init, g_init,
                        sigma_init, w_init], [v_init, v_final],
                        args=(other_variables, F, solver,), full_output=True)

                    r_soln = soln_v[1][0]
                    if r_soln < r_break:
                        print "-"*100
                        print "Breaking v loop"
                        print "-"*100
                        break

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

                    r[u_coord + 1, v_coord + 1] = soln_v[1][0]
                    f[u_coord + 1, v_coord + 1] = soln_v[1][1]
                    g[u_coord + 1, v_coord + 1] = soln_v[1][2]
                    sigma[u_coord + 1, v_coord + 1] = soln_v[1][3]
                    w[u_coord + 1, v_coord + 1] = soln_v[1][4]

    def derivative(self, var_symbol, direction, coords=None, order=1, der=1):
        U = self.U; V = self.V

        r = self.r
        f = self.f
        g = self.g
        sigma = self.sigma
        d = self.d
        w = self.w
        z = self.z

        variables = {'r' : r, 'f' : f, 'g' : g, 'sigma' : sigma, 'd' : d, 'w' :
            w, 'z' : z}

        var = variables[var_symbol]

        if coords!=None:
            u_coord = coords[0]; v_coord = coords[1]

            if direction=='v':

                if v_coord==0:
                    print "Derivative in the direction of v at v=0 is not \
                        known."
                    print "Please provide boundary data for the derivative."
                    sys.exit(1)

                if v_coord <=3:
                    tck, fp, ierr, msg = splrep(V[u_coord, :4], var[u_coord,
                        :4], k=1, full_output=True)

                    if ierr > 0:
                        print "Error in spline representation used for \
                            derivative calculation"
                        sys.exit(1)

                    return splev(V[u_coord, v_coord], tck, der=der)

                tck, fp, ierr, msg = splrep(V[u_coord, v_coord-4:v_coord],
                    var[u_coord, v_coord-4:v_coord], k=order, full_output=True)

                if ierr > 0:
                    print "Error in spline representation used for derivative \
                        calculation"
                    sys.exit(1)

                return splev(V[u_coord, v_coord], tck, der=der)

            if direction=='u':

                if u_coord==0:
                    print "Derivative in the direction of u at u=0 is not \
                        known."
                    print "Please provide boundary data for the derivative."
                    sys.exit(1)

                if u_coord <=3:
                    tck, fp, ierr, msg = splrep(U[:4, v_coord], var[u_coord,
                        :4], k=1, full_output=True)

                    if ierr > 0:
                        print "Error in spline representation used for \
                            derivative calculation"
                        sys.exit(1)

                    return splev(U[u_coord, v_coord], tck, der=der)

                tck, fp, ierr, msg = splrep(U[u_coord-4:u_coord, v_coord],
                    var[u_coord-4:u_coord, v_coord], k=order, full_output=True)

                if ierr > 0:
                    print "Error in spline representation used for derivative \
                        calculation"
                    sys.exit(1)

                return splev(U[u_coord, v_coord], tck, der=der)

        elif coords==None:

            if direction=='v':
                dvar_dv = np.zeros([self.Nu, self.Nv])
                for u_coord in xrange(self.Nu):
                    tck, fp, ierr, msg = splrep(V[u_coord, :], var[u_coord, :],
                        k=order, full_output=True)

                    if ierr > 0:
                        print "Error in spline representation used for \
                            derivative calculation"
                        sys.exit(1)

                    dvar_dv[u_coord, :] = splev(V[u_coord, :], tck, der=der)

                return dvar_dv

            if direction=='u':
                dvar_du = np.zeros([self.Nu, self.Nv])
                for v_coord in xrange(self.Nv):
                    tck, fp, ierr, msg = splrep(U[:, v_coord], var[:, v_coord],
                        k=order, full_output=True)

                    if ierr > 0:
                        print "Error in spline representation used for \
                            derivative calculation"
                        sys.exit(1)

                    dvar_du[:, v_coord] = splev(U[:, v_coord], tck, der=der)

                return dvar_du

    def constraint(self, order=1):
        r = self.r
        f = self.f
        w = self.w
        alpha = self.alpha

        df_du = self.derivative('f', 'u', order=order)
        dsigma_du = self.derivative('sigma', 'u', order=order)

        if alpha==0:
            C1 = df_du - 2*f*dsigma_du

            return C1

        if self.FullScalarFieldEvolution:
            dw_du = self.derivative('w', 'u', order=order)

            C1 = df_du - 2*f*dsigma_du + alpha*(w**2/4. + dw_du/2. -
                w*dsigma_du)/r
            return C1
        else:
            dsigma_duu = self.derivative('sigma', 'u', order=order, der=2)

            C1 = df_du - 2*f*dsigma_du + alpha*(dsigma_duu - dsigma_du**2)/r
            return C1

    def Ricci_scalar(self, order=1):
        r = self.r
        f = self.f
        g = self.g
        sigma = self.sigma

        df_dv = self.derivative('f', 'v', order=order)
        dd_du = self.derivative('d', 'u', order=order)

        R = -np.exp(-2*sigma)*(16*r*df_dv + 8*r**2*dd_du + 8*f*g +
                2*np.exp(2*sigma))/r**2

        return R

    def quantum_stress_energy_tensor(self, i, j, order=1):
        alpha = self.alpha
        r = self.r
        f = self.f
        g = self.g
        sigma = self.sigma
        d = self.d
        w = self.w
        z = self.z

        if i=='u' and j=='u':

            dw_du = self.derivative('w', 'u', order=order)
            dsigma_du = self.derivative('sigma', 'u', order=order)

            return (w**2/4. + dw_du/2. - w*dsigma_du)/(4*np.pi*r**2)

        if i=='v' and j=='v':

            dz_dv = self.derivative('z', 'v', order=order)

            return (z**2/4. + dz_dv/2. - z*d)/(4*np.pi*r**2)

        if (i=='u' and j=='v') or (i=='v' and j=='u'):

            return -(f*g + np.exp(2*sigma)/4.)/(4*np.pi*r**2*(r**2 -
                alpha))

    def save_data(self, filename):
        U = self.U; V = self.V
        source_array = self.classical_source(V[0, :])
        scalar_field_array = self.scalar_field(V[0, :])
        alpha = self.alpha
        FullScalarFieldEvolution = self.FullScalarFieldEvolution
        r = self.r
        f = self.f
        g = self.g
        sigma = self.sigma
        d = self.d
        w = self.w
        z = self.z

        variables = [U, V, source_array, scalar_field_array, alpha,
            FullScalarFieldEvolution, r, f, g, sigma, d, w, z]
        variable_names = ['U', 'V', 'source_array', 'scalar_field_array',
            'alpha', 'FullScalarFieldEvolution', 'r', 'f', 'g', 'sigma','d',
            'w', 'z']

        datafile = h5py.File(filename, 'w')
        for var, i in enumerate(variable_names):
            datafile[i] = variables[var]
        datafile.close()
