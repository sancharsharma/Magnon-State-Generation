

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.optimize as scopt

import MagnonTransmon as MagTr
import Simulation as Sim
import GenerationSequence as Gen
from importlib import reload
import h5py

reload(MagTr)
reload(Sim)
reload(Gen)

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

#### Definition of states
fock = lambda n: qt.fock(Nm,n)
fock_sup = lambda n: (qt.fock(Nm,0) + qt.fock(Nm,n)).unit()
coh = lambda pos: qt.coherent(Nm,pos,method='analytic')
cat_even = lambda Cpos: (coh(Cpos/2) + coh(-Cpos/2)).unit()  # Function to create an even cat state
cat_odd = lambda Cpos: (coh(Cpos/2) - coh(-Cpos/2)).unit()  # Function to create an odd cat state


#### Target states in the form of [name, name]
States = []
States.append(["MediumEvenCat",cat_even(4)])
States.append(["LargeEvenCat",cat_even(5)])
States.append(["MediumOddCat",cat_odd(4)])
States.append(["LargeOddCat",cat_odd(5)])
States.append(["MediumFock",fock(6)])
States.append(["LargeFock",fock(10)])
States.append(["MediumSuperpos",fock_sup(6)])
States.append(["LargeSuperpos",fock_sup(10)])

#At the end, each entry of States will be of the form ['name',target,achieved,fidelity]
no_targets = len(States)
for i in range(no_targets):
	name = States[i][0]
	target = States[i][1]

	System = MagTr.MagTr(System_pars)

	# Seq is the sequence of operations to be performed. Psi_expect_target stores the expected states after each operation ignoring dissipation.
	Seq,Psi_expect_target = Gen.Sequence(target,System_pars)
	if Seq == []:
		States[i].append(qt.fock(Nm,0))
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
	ang_corrs = scopt.minimize_scalar(lambda theta: -qt.fidelity( fin_magnon_state, mag_rot(theta)*target ) )
	fid,ang = -ang_corrs['fun'],ang_corrs['x']
	
	corrected_state = mag_rot(-ang)*fin_magnon_state*mag_rot(ang)

	States[i].append(corrected_state)
	States[i].append(fid)

	del System

## Modified qutip's plot_wigner function
def plot_wigner2(rho, fig=None, ax=None):

	if not fig and not ax:
		fig, ax = plt.subplots(1, 1)

	alpha_max=5  ## Adjustable

	if qt.isket(rho):
		rho = qt.ket2dm(rho)

	vec = np.linspace(-alpha_max, alpha_max, 500)
	W = qt.wigner(rho, vec, vec, method='iterative')
	
	# Our convention W_{mag} is different than qutip's convention W_Q: W_{mag}(M_x,M_y) = 2\pi W_Q(M_x/sqrt(2),-M_y/sqrt(2))
	
	W_scaled = (2*np.pi)*W
	xvec = np.sqrt(2)*vec
	yvec = -xvec
	wlim = abs(W_scaled).max()

	cmap = mpl.cm.get_cmap('RdBu')

	cf = ax.contourf(xvec, yvec, W_scaled, 300,
                         norm=mpl.colors.Normalize(-wlim, wlim), cmap=cmap)

	xtickpos = int(round(1.1*alpha_max))
	axis_ticks = [-xtickpos,0,xtickpos]
	ax.set_xticks(axis_ticks)
	ax.set_yticks(axis_ticks)

	if (W_scaled.min() > -0.05): # If the minimum is too close to 0, we don't need to put it on the colobar
		tickpos = [0.95*W_scaled.max(),0]
	else:
		tickpos = [0.95*W_scaled.max(),0,0.95*W_scaled.min()]

	cbar = fig.colorbar(cf, ax=ax, ticks=tickpos)
	cbar.ax.set_yticklabels(['{v:.2f}'.format(v=tp) for tp in tickpos])

	return fig, ax

##### Code to save every figure separately in a file. It requires a folder named Figures in the directory of working.
#
#plt.rcParams.update({'font.size': 22})
#
#for i in range(no_targets):
#	name = States[i][0]
#	target = States[i][1]
#	achieved = States[i][2]
#
#	fig_tar, ax_tar = plot_wigner2(target) 
#	fig_tar.savefig('Figures/' + name + '_target.png',dpi = 400,bbox_inches = 'tight')
#
#	fig_res, ax_res = plot_wigner2(achieved)
#	fig_res.savefig('Figures/' + name + '_achieved.png',dpi = 400,bbox_inches = 'tight')
#
#	del fig_tar,ax_tar,fig_res,ax_res



##### Code to plot everything in one figure
plt.rcParams.update({'font.size': 22})
figure = plt.figure(figsize=(18,8*no_targets))

axes_target = ['None']*(no_targets)
axes_res = ['None']*(no_targets)

for i in range(no_targets):
	name = States[i][0]
	target = States[i][1]
	achieved = States[i][2]
	axes_target[i] = figure.add_subplot(no_targets,2,2*i+1)
	axes_res[i] = figure.add_subplot(no_targets,2,2*i+2)
	plot_wigner2(target, fig=figure, ax=axes_target[i])
	plot_wigner2(achieved, fig=figure, ax=axes_res[i])

	axes_target[i].set_xlabel(r'$\frac{M_x}{{\cal M}_{\rm ZPF}}$',fontsize = 22)
	axes_target[i].set_ylabel(r'$\frac{M_y}{{\cal M}_{\rm ZPF}}$',fontsize = 22)

	axes_res[i].set_xlabel(r'$\frac{M_x}{{\cal M}_{\rm ZPF}}$',fontsize = 22)
	axes_res[i].set_ylabel(r'$\frac{M_y}{{\cal M}_{\rm ZPF}}$',fontsize = 22)


axes_target[0].set_title('States',fontsize=24)
axes_res[0].set_title('Results',fontsize=24)



print('Program finished.')


