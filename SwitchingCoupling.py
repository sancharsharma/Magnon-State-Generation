import numpy as np
import qutip as qt
from cmath import polar
import scipy.optimize as scopt

import MagnonTransmon as MagTr
from importlib import reload
reload(MagTr)

ZeroMat = lambda dim: np.zeros(dim) + 1j*np.zeros(dim)
##### Switching the coupling

### Parameters
def T_Params(psi,j,pars):

	Nm = pars['Size_m']
	Nq = pars['Size_q']
	delta = pars['omega_m'] - pars['omega_q']

	TLS = pars.get('TLS',0)

	Fock_0 = qt.tensor( qt.fock(Nm,j+1),qt.fock(Nq,0) )
	Fock_1 = qt.tensor( qt.fock(Nm,j),qt.fock(Nq,1) )
	Psi_0,phi_0 = polar(Fock_0.overlap(psi)) # Projection onto \ket{j,0}
	Psi_1,phi_1 = polar(Fock_1.overlap(psi)) # Projection onto \ket{j-1,1}
	G_0,x_0 = polar(pars['g_mq'] * np.sqrt(j+1))
	sat = 1

	if (Psi_0 < 0.1):
		t_off = 0
		t_on = 0
	else:
		t_off_g = ((phi_1 - phi_0 - x_0 - np.pi/2) % (2*np.pi))/delta 
		t_on_g = np.arctan(Psi_0/(Psi_1+1e-12))/G_0

		if (j==0 or Nq==2 or TLS != 0):
			Geff = np.sqrt(G_0**2 + delta**2/4)
			Mat_off_2 = np.array([
				[-delta/2,G_0],
				[G_0,delta/2]
			])/Geff

			Mat_off = lambda t : np.cos(Geff*t)*np.identity(2) - 1j*np.sin(Geff*t)*Mat_off_2
			Mat_on = lambda t : np.cos(G_0*t)*np.identity(2) - 1j*np.sin(G_0*t)*np.array([[0,1],[1,0]])

			to_opt = lambda t: np.array([Psi_1*np.exp(-1j*phi_1-1j*x_0),Psi_0*np.exp(-1j*phi_0)]).dot(Mat_off(t[1])).dot(Mat_on(t[0])).dot(np.array([0,np.exp(1j*x_0)]))

		else:
			A = pars['alpha']
			Size = np.min([Nq,j+2])
			
			Psi_herm = ZeroMat(Size)
			s_vec = np.arange(Size)
			for s in s_vec:
				Fock_s = qt.tensor( qt.fock(Nm,j+1-s),qt.fock(Nq,s) )
				Psi_herm[s] = psi.overlap(Fock_s) # Projection onto \ket{j+1-s,s}
			
			M_on = ZeroMat((Size,Size))
			for s in s_vec:
				M_on[s,s] = -(A/2)*s*(s-1) if s>1 else 0
				if (s<=Size-2):
					M_on[s,s+1] = pars['g_mq'] * np.sqrt(s+1) * np.sqrt(j+1-s)
				if (s>=1):
					M_on[s,s-1] = pars['g_mq'] * np.sqrt(s) * np.sqrt(j+2-s)

			M_off = M_on - np.diagflat(delta*s_vec)
			
			# M = V*diagflat(Lambda)*V^(-1)
			Lambda_off,V_off = np.linalg.eig(M_off)  
			Lambda_on,V_on = np.linalg.eig(M_on)  

			v_Psi = np.diagflat(Psi_herm.dot(V_off))
			v_1 = np.diagflat(V_on.conj()[0])
			V_fin = v_Psi.dot(V_off.T.conj()).dot(V_on).dot(v_1)

			on_vec = lambda t: np.exp(-1j*Lambda_on*t)
			off_vec = lambda t: np.exp(-1j*Lambda_off*t)
			
			to_opt = lambda t: off_vec(t[1]).dot(V_fin).dot(on_vec(t[0]))
	
			
		compl_to_vect = lambda z: [np.real(z),np.imag(z)]
		to_opt_vec = lambda t: compl_to_vect(to_opt(t))
		#sols = scopt.fsolve(to_opt_vec,[t_on_g,t_off_g],full_output=1)
		#t_on = sols[0][0]
		#t_off = sols[0][1]

		sol = scopt.minimize(lambda t: np.abs(to_opt(t)),[t_on_g,t_off_g])
		t_on,t_off = sol['x']

		sat = 1 if np.abs(sol['fun']) < 0.01 else 0

		#if (sols[2]!=1 or t_on<0 or t_off<0 or t_on>np.pi/(2*G_0) or t_off>4*np.pi/delta):
		if (t_on<0 or t_off<0 or t_on>np.pi/(2*G_0) or t_off>4*np.pi/delta):
			t_on = t_on_g
			t_off = t_off_g
			sat = 0
		
	return t_on, t_off, sat


#### Non-dissipative waiting operation
def Op_T(t_on,t_off,pars):
	
	lpars = pars.copy()
	lpars['RotFrame'] = 0

	sys = MagTr.MagTr(lpars)
	H_off = sys.Hams[0]
	delta = lpars['omega_m'] - lpars['omega_q']
	H_on = H_off + delta*sys.n_q
	T1=(-1j*H_off*t_off).expm()
	T2=(-1j*H_on*t_on).expm()
	return T1*T2


