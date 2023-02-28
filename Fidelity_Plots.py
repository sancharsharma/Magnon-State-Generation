

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import minimize_scalar
import h5py

import MagnonTransmon as MagTr
import Simulation as Sim
import GenerationSequence as Gen
import Classes_States as Cls
from importlib import reload

reload(MagTr)
reload(Sim)
reload(Gen)
reload(Cls)

Nm = 19
Nq = 6

##### Definition of system parameters 

#All values are in GHz or nanosecond
#Notice that we have a lot of dissipation!

omega_m = 2*np.pi*6  # Magnon frequency
omega_q = 2*np.pi*5  # Frequency of 0-1 transition in transmon
kappa_m = 1e-4*omega_m  # Magnon dissipation
decay_q = 1e-4*omega_q  # Transmon decay rate
deph_q = 1e-4*omega_q  # Transmon dephasing rate
A = 2*np.pi*0.3  # Transmon's anharmonicity parameter
coup = 2*np.pi*25e-3  # Coupling between magnon and transmon

N_tau = 3  # Multiplier for gate time
t_g = N_tau*np.pi/A  # Gate time

System_pars = {
	'Size_m' : Nm,  # max no of magnons = 'Size_m' - 1
	'Size_q' : Nq,  # max no of transmons = 'Size_q' - 1
	'omega_m' : omega_m,
	'omega_q' : omega_q,
	'kappa_m' : kappa_m,
	'decay_q' : decay_q,
	'deph_q' : deph_q,
	'g_mq' : coup,
	'alpha' : A,
	'gate_time' : N_tau,  #multiplier of pi/A
	'RotFrame' : 1, # If defined and \ne 0, (exact) rotating frame is used to speed up simulation
	'TLS' : 0 # If defined and \ne 0, the sequence is generated assuming a two-level approximation
	}

#### What do we want to plot? Defined CatsEven, CatsOdd, Focks, FocksSup. See Classes_States.py for definitions or changes
cls = Cls.FocksSup(Nm)  

#### Data storage
fidelities_opt = []

#### fidelities_opt will be an array with each entry as [par,fid] (see Classes_States.py for definition of pars for each class)
for ii in range(cls.nmax):
	
	target = cls.states[ii]
	System = MagTr.MagTr(System_pars)

	# Seq is the sequence of operations to be performed. Psi_expect_target stores the expected states after each operation ignoring dissipation.
	Seq,Psi_expect_target = Gen.Sequence(target,System_pars)
	if Seq == []:  # Ignore the states very close to the ground state
		continue

	# coup_pulse is Delta(t), gate_pulse is eps(t) in rotated frame, Prot_time is the total protocol time.
	coup_pulse,gate_pulse,Prot_time = Gen.Pulses(Seq,System_pars)

	System.freq_mod(coup_pulse)
	System.mw_pulse(gate_pulse)

	Simulation_pars = {
		'init_state' : System.vac,
		'time_final' : Prot_time
	}

	#System.Linds=[]  # Uncomment to remove dissipation

	result,vars,times = Sim.Simulate(System,Simulation_pars)  # Simulate the dynamics. See the file Simulation.py for explanation of output
	
	# Converting to lab frame if rotating frame is used
	rot_Op = (-1j*omega_q*Prot_time*System.n_q - 1j*omega_m*Prot_time*System.n_m).expm() if System_pars.get('RotFrame',0) != 0 else System.Id
	fin_states = rot_Op*result.states[-1]*rot_Op.dag()
	fin_magnon_state = (fin_states.ptrace(0)).unit()  # Trace out transmon

	# The resulting state can be a rotated version of the target, which we correct for.
	mag_rot = lambda theta: (-1j*theta*qt.num(Nm)).expm()
	ang_corrs = minimize_scalar(lambda theta: -qt.fidelity( fin_magnon_state, mag_rot(theta)*target ) )
	fid,ang = -ang_corrs['fun'],ang_corrs['x']
	
	corrected_state = mag_rot(-ang)*fin_magnon_state*mag_rot(ang)

	fidelities_opt.append([cls.pars[ii],fid])

	print('-----XXXX-----XXXX-----XXX-----')
	print('')
	print('we are done with count '+ str(ii+1) + '/' + str(cls.nmax))
	print('')
	print('-----XXXX-----XXXX-----XXX-----')

	del System
	del target

print('Waiting for data to be stored...')

h5file = h5py.File('DataStateGeneration.hdf5', 'a')

#Overwrite previous results
if cls.data_label in h5file.keys():
    del h5file[cls.data_label]
#Add the new results for the fidelities
h5file.create_dataset(cls.data_label, data=fidelities_opt)

h5file.close()

print('Data stored. Program finished.')
