
import qutip as qt
import numpy as np

def Simulate(Sys,pars):

	ts = 0.01 * 2*np.pi / Sys.freq_scale  # Nanoseconds 
	tf = pars['time_final']  # Nanoseconds

	times = np.linspace(0, tf, round(tf / ts))
	
	list_of_vars = list(Sys.AvgsDict.keys())

	# Creating an array for input to mesolve
	Avgs = []
	for variable in list_of_vars:
		Avgs.append(Sys.AvgsDict[variable])

	## And we are ready
	result = qt.mesolve(Sys.Hams, pars['init_state'], times, Sys.Linds, Avgs, progress_bar=True, options = qt.Options(store_states = True)) ## Can also do store_states=True or store_final_state=True
	
	# Create a dictionary of all the expectation values calculated. vars['Q_11'], e.g.
	vars = {}
	for i in range( 0,len(list_of_vars) ):
	    vars[list_of_vars[i]] = result.expect[i]
	
	return result,vars,times


	
