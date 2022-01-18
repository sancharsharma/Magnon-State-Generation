
import numpy as np
import qutip as qt
import scipy.linalg as lin

import MagnonTransmon as MagTr
import SwitchingCoupling as Coup
import Gate_Functions as Gates
from importlib import reload
reload(MagTr)
reload(Coup)
reload(Gates)


ZeroMat = lambda dim: np.zeros(dim) + 1j*np.zeros(dim)

#### Convert the sequence to functions for qutip
#### Seq consists of terms ['gate',pulse_fun], ['transfer',[t_on,t_off]]. It is supposed to be generated using 'Sequence' function 
def Pulses(Seq, pars):

	om_q = pars['omega_q']
	Delta_0 = pars['omega_m'] - pars['omega_q']

	N_tau = pars['gate_time']  
	A = pars['alpha']  # anharmonicity
	t_g = N_tau*np.pi/A  # gate time in nanoseconds
	fr = pars.get('RotFrame',0)  # whether rotating frame is used


	# The function Delta(t) is a piecewise function. To input numpy.piecewise, we need a set of conditions, here Transfers_conds, and the values when those conditions are satisfied, here Transfers_funcs. Transfer_funcs will look like [Delta_0,0,Delta_0,0,...] s.t. Delta(t) = Delta_0 for conds[0]<t<conds[1] (excitation transfer time), =0 for conds[1]<t<conds[2] (gate time) and so on. Similar holds for Gates except that Delta_0 is replaced by functions and there might be consecutive gates.
	Transfers_conds = [] 
	Transfers_funcs = []
	Gates_conds = []
	Gates_funcs = []

	# Create a delayed and possibly rotated version of the input function
	def gen_fun(fun,start):
		if (fr==0): 
			return lambda t: fun(t-start)  # Lab frame
		else:
			return lambda t: fun(t-start)*np.exp(1j*om_q*t)  # Rotating frame
			
	Time = 0  # Variable to keep track of time
	
	for el_op in Seq:
		if (el_op[0] == 'transfer'):
			# el_op[1] looks like (t_on,t_off)
			Time_next = Time + el_op[1][0] 
			Transfers_conds.append(Time)
			Transfers_conds.append(Time_next)
			Transfers_funcs.append(Delta_0)
			Transfers_funcs.append(0)
			Time = Time_next + el_op[1][1]
		elif (el_op[0] == 'Q_gate' or el_op[0] == 'L_gate'):
			Gates_conds.append(Time)
			Gates_conds.append(Time+t_g)
			Gates_funcs.append(gen_fun(el_op[1],Time))
			Gates_funcs.append(0)
			Time = Time + t_g
		else:
			raise ValueError('Unrecognizable Sequence head. Use -transfer-, -Q_gate- or -L_gate-')

	# The second argument 'args' is for qutip syntaxing
	coup_pulse = lambda t,args: np.piecewise(t,t>np.array(Transfers_conds),Transfers_funcs)  # Delta(t)
	gate_pulse = lambda t,args: np.piecewise(t+0j,t>np.array(Gates_conds),Gates_funcs)  # \tilde{\epsilon}(t)

	return coup_pulse,gate_pulse,Time

#### Generating the sequence function. 
def Sequence(target,pars):
	
	# target.shape[0] should equal pars['Size_m'].
	Nm = pars['Size_m']  
	Nq = pars['Size_q']
	A = pars['alpha']
	N_tau = pars['gate_time']
	t_g = N_tau*np.pi/A

	# Create a local copy of pars to get the Hamiltonian in the lab frame.
	lpars = pars.copy()
	lpars['RotFrame'] = 0
	sys = MagTr.MagTr(lpars)
	Phase = (-1j*sys.Hams[0]*t_g).expm()

	Seq = []  # To be created
	Psi_temp = qt.tensor(target,qt.fock(Nq,0))  # Keeps track of the wave-function during the process starting from final state reaching down to vacuum

	Psi_vec = ['none']*Nm  # To store all intermediate wave-functions
	Psi_vec[Nm-1] = [Psi_temp]*3 
	
	# At j-th step, we remove j+1 excitation, i.e. (j+1,0), (j,1), (j-1,2) [(a,b) stands for 'a' magnons and 'b' transmons]
	for j in range(Nm-2,-1,-1): 
		
		# Remove (j+1,0)
		sat = 0  # flag to set when (j+1,0) is removed
		while (sat == 0):
			t_on, t_off, sat = Coup.T_Params(Psi_temp,j,pars)
			if (t_on != 0 or t_off != 0):
				Seq.append(['transfer',[t_on,t_off]])
				Psi_temp = Coup.Op_T(t_on,t_off,pars).dag()*Psi_temp
		Psi_vec[j] = [Psi_temp]

		# Remove (j,1)
		sol = Gates.Q_Params(Psi_temp,j,pars)
		if (sol != 'none'):  # sol = 'none' implies no gate is required, i.e. (j,1) is already small
			Seq.append(['Q_gate',sol[0]])
			U_Op = qt.tensor(qt.qeye(Nm), qt.Qobj(sol[1]))  # Corrections of the gate in the rotating frame
			Op_Q = Phase*U_Op  # Correct for phases
			Psi_temp = Op_Q.dag()*Psi_temp
		Psi_vec[j].insert(0,Psi_temp)
		
		# Remove (j-1,2)
		sol = Gates.L_Params(Psi_temp,j,pars)
		if (sol != 'none'):
			Seq.append(['L_gate',sol[0]])
			U_Op = qt.tensor(qt.qeye(Nm), qt.Qobj(sol[1]))
			Op_L = Phase*U_Op
			Psi_temp = Op_L.dag()*Psi_temp
		Psi_vec[j].insert(0,Psi_temp)
	
	Seq.reverse()

	return Seq,Psi_vec


