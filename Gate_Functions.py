import numpy as np
import qutip as qt
import scipy.linalg as lin
from cmath import polar

import MagnonTransmon as MagTr
import ErrorCorrection_Gates as Corr
from importlib import reload
reload(MagTr)
reload(Corr)

ZeroMat = lambda dim: np.zeros(dim) + 1j*np.zeros(dim)
def Q_Params(psi,j,pars):

	Nm = pars['Size_m']
	Nq = pars['Size_q']
	omega_q = pars['omega_q']
	A = pars['alpha']
	N_tau = pars['gate_time']
	t_g = N_tau*np.pi/A

	# Dressed Frequency
	g = pars['g_mq']
	omega_m = pars['omega_m']
	omega_q = omega_q - g**2/(omega_m - omega_q)
	
	TLS = pars.get('TLS',0)
	
	if ((TLS != 0) or (Nq == 2)):
		order = 0  # Zeroth order of Magnus correction means no corrections
	else:
		order = 1 # First order corrections

	Fock_0 = qt.tensor( qt.fock(Nm,j),qt.fock(Nq,0) )
	Fock_1 = qt.tensor( qt.fock(Nm,j),qt.fock(Nq,1) )
	Psi_0,phi_0 = polar(Fock_0.overlap(psi)) # Projection onto \ket{j,0}
	Psi_1,phi_1 = polar(Fock_1.overlap(psi)) # Projection onto \ket{j,1}

	if (Psi_1 < 0.1):
		return 'none'  # No gate required
	else:

		theta = np.arctan(Psi_1/(Psi_0 + 1e-12))
		indices,eps_mu = Corr.eps_corr(theta,A,N_tau,order)  # corrected pulse parameters
		freqs = (2*indices+1)*np.pi/t_g
		eps = lambda t: np.dot(eps_mu,np.sin(freqs*t))  # \epsilon(t), pulse after phase removal

		#  The evolution matrix in the transmon space
		def N(t):
			res = ZeroMat((Nq,Nq))
			for s in range(Nq-1):
				res[s,s+1] = np.exp(1j*A*s*t)*np.sqrt(s+1)
				res[s+1,s] = np.exp(-1j*A*s*t)*np.sqrt(s+1)

			return eps(t)*res
		
		Ueff_wophase = Corr.timeexp(N,t_g,max(freqs))  # Computes Texp[1j*\int_0^{t_g} N] numerically

		V,_,Wh = lin.svd(Ueff_wophase[0:2,0:2])
		Ueff_01 = V.dot(Wh)  # This gives a unitary approximation to 2x2 block between 0 and 1 states
		#Here, we can potentially check how close atan(Ueff_01[0,0]/Ueff_1[0,1]) is to theta.
		#print(j,np.arccos(np.abs(Ueff_01[0,0])),theta)

		phi_d = (np.angle(Ueff_01[0,0]) - np.angle(Ueff_01[1,1]))/2
		phi_od = (np.angle(Ueff_01[0,1]) - np.angle(Ueff_01[1,0]))/2

		phi = (omega_q*t_g - phi_0 + phi_1 + phi_d + phi_od - np.pi/2) % (2*np.pi)  # The phase of the gate
		U_phase = np.diagflat(np.exp(1j*phi*np.linspace(0,Nq-1,Nq)))
		
		tilde_eps = lambda t: -np.dot(eps_mu,np.sin(freqs*t))*np.exp(1j*phi)*np.exp(-1j*omega_q*t)
		Ueff = U_phase.dot(Ueff_wophase).dot(U_phase.conj())

	return tilde_eps,Ueff

def L_Params(psi,j,pars):
	
	Nm = pars['Size_m']
	Nq = pars['Size_q']
	omega_q = pars['omega_q']
	A = pars['alpha']
	N_tau = pars['gate_time']
	t_g = N_tau*np.pi/A

	# Dressed frequency
	g = pars['g_mq']
	omega_m = pars['omega_m']
	omega_q = omega_q - g**2/(omega_m - omega_q)

	TLS = pars.get('TLS',0)
	
	if ((TLS != 0) or (Nq == 2)):
		order = 0  # Zeroth order of Magnus correction means no corrections
	else:
		order = 1  # First order correction


	if (j==0 or Nq==2 or TLS!=0):
		return 'none'

	Fock_1 = qt.tensor( qt.fock(Nm,j-1),qt.fock(Nq,1) )
	Fock_2 = qt.tensor( qt.fock(Nm,j-1),qt.fock(Nq,2) )
	Psi_1,phi_1 = polar(Fock_1.overlap(psi)) # Projection onto \ket{j-1,1}
	Psi_2,phi_2 = polar(Fock_2.overlap(psi)) # Projection onto \ket{j-1,2}

	if (Psi_2 < 0.1):
		return 'none'  # No gate required
	else:

		theta = np.arctan(Psi_2/(Psi_1 + 1e-12))
		indices,eps_mu = Corr.eps_corr(theta,A,N_tau,order)  # corrected pulse parameters
		freqs = (2*indices+1)*np.pi/t_g
		eps = lambda t: np.dot(eps_mu,np.sin(freqs*t))  # \epsilon(t), pulse after phase removal


		#  The evolution matrix in the transmon space
		def N(t):
			res = ZeroMat((Nq,Nq))
			for s in range(Nq-1):
				res[s,s+1] = np.exp(1j*A*(s-1)*t)*np.sqrt(s+1)
				res[s+1,s] = np.exp(-1j*A*(s-1)*t)*np.sqrt(s+1)

			return eps(t)*res/np.sqrt(2)
		
		Ueff_wophase = Corr.timeexp(N,t_g,max(freqs))  # Computes Texp[1j*\int_0^{t_g} N]

		V,_,Wh = lin.svd(Ueff_wophase[1:3,1:3])
		Ueff_12 = V.dot(Wh)  # This gives a unitary approximation to 2x2 block between 1 and 2 states
		#Here, we can check how close atan(Ueff_12[0,0]/Ueff_12[0,1]) is to theta. If it is too far, then perhaps second order corrections should be included.
		#print(j,np.arccos(np.abs(Ueff_12[0,0])),theta)

		phi_d = (np.angle(Ueff_12[0,0]) - np.angle(Ueff_12[1,1]))/2
		phi_od = (np.angle(Ueff_12[0,1]) - np.angle(Ueff_12[1,0]))/2

		phi = ((omega_q-A)*t_g - phi_1 + phi_2 + phi_d + phi_od - np.pi/2) % (2*np.pi)  # The phase of the gate
		
		U_phase = np.diagflat(np.exp(1j*phi*np.linspace(0,Nq-1,Nq)))
		
		tilde_eps = lambda t: -np.dot(eps_mu,np.sin(freqs*t))*np.exp(1j*phi)*np.exp(-1j*(omega_q-A)*t)/np.sqrt(2)
		Ueff = U_phase.dot(Ueff_wophase).dot(U_phase.conj())

	return tilde_eps,Ueff


