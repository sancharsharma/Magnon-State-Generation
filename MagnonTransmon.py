
import numpy as np
import qutip as qt



################## This class is for a magnon coupled to a transmon
class MagTr:

	##### System params
	
	def __init__(self, pars):
		
		# Size of Hilbert spaces
		Nm = pars['Size_m']
		Nq = pars['Size_q']
	
		# Operators
		Id = qt.tensor(qt.qeye(Nm), qt.qeye(Nq))
		m = qt.tensor(qt.destroy(Nm), qt.qeye(Nq))
		q = qt.tensor(qt.qeye(Nm), qt.destroy(Nq))
		n_m = m.dag()*m
		n_q = q.dag()*q
	
		# All pars are in units of GHz or ns
		
		# Uncoupled Hamiltonian   
		H_m = pars['omega_m'] * n_m
		H_q = pars['omega_q'] * n_q
		H_anh = -pars['alpha']*n_q*(n_q-1)/2
		H_0 = H_q + H_m + H_anh

		# Coupling
		H_mq = pars['g_mq'] * m*q.dag()
		H_qm = np.conjugate(pars['g_mq']) * q*m.dag()

		# Total Hamiltonian
		H_sys = H_0 + H_mq + H_qm

		vac = H_sys.eigenstates()[1][0]  # Ground state

		Hams = []
		# The rotating frame is defined w.r.t H_m+H_q (not including the anharmonic component). If pars['RotFrame'] is defined and \ne 0, then rotating frame is used.
		if (pars.get('RotFrame',0) == 0):
			Hams.append(H_sys)
			
			freq_scale = max([pars['omega_m'],pars['omega_q']]) # Decides the simulation time step
		else:
			Hams.append(H_anh)
			Hams.append([H_mq, 
				lambda t,args: np.exp( 1j*(pars['omega_q']-pars['omega_m'])*t ) ])
			Hams.append([H_qm, 
				lambda t,args: np.exp( 1j*(pars['omega_m']-pars['omega_q'])*t ) ])

			freq_scale = max(np.abs( [ pars['alpha'] , pars['g_mq'] , pars['kappa_m'] , pars['decay_q'] , pars['deph_q'] , pars['omega_q']-pars['omega_m'] ] ))
			

		#### Lindblad operators
		Linds = []
		Linds.append(np.sqrt(pars['kappa_m']) * m)  
		Linds.append(np.sqrt(pars['decay_q']) * q)
		Linds.append(np.sqrt(pars['deph_q']) * n_q)
			
		##### We ask mesolve to store these averages. To add a new observable, choose a 'name' freely and then add the line AvgsDict['name'] = Operator. The 'name' will be used for plotting.
		
		AvgsDict = {}  # Dictionary
	
		AvgsDict['num_m'] = n_m
		
		AvgsDict['Q_11'] = qt.tensor( 
			qt.qeye(Nm) , qt.fock(Nq,1) * qt.fock(Nq,1).dag())  # |1><1|
		AvgsDict['Q_00'] = qt.tensor( 
			qt.qeye(Nm) , qt.fock(Nq,0) * qt.fock(Nq,0).dag())  # |0><0|
		AvgsDict['Q_10'] = qt.tensor( 
			qt.qeye(Nm) , qt.fock(Nq,1) * qt.fock(Nq,0).dag())  # |1><0|

		if (Nq > 2):
			AvgsDict['Q_comp'] = qt.tensor( qt.qeye(Nm),
				qt.fock_dm(Nq,0) + qt.fock_dm(Nq,1) + qt.fock_dm(Nq,2))  # Projection onto the computational space for higher dimensional transmons

	
		### All these variables can be read outside the scope
		self.Nm = Nm
		self.Nq = Nq
		self.Id = Id
		self.m = m
		self.q = q
		self.n_m = n_m
		self.n_q = n_q

		self.vac = vac
		self.freq_scale = freq_scale

		self.H_0 = H_0
		self.Hams = Hams
		self.AvgsDict = AvgsDict
		self.Linds = Linds

	### The function to modulate qubit frequency as a function of time. Qutip requires shape to have two arguments, the first being time and second being a set of arguments.
	def freq_mod(self,shape):
		self.Hams.append([self.n_q,shape])
	
	### Transmon pulse
	def mw_pulse(self,shape):
		# shape is a complex function
		self.Hams.append([self.q.dag(), shape])
		self.Hams.append([self.q, 
			lambda t,args: np.conj(shape(t,args)) ])
	
	###### End of MagTr
	
