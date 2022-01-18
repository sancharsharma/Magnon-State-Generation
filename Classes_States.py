import qutip as qt
import numpy as np


class CatsEven:
	def __init__(self, Nm, nmax = 300): #nmax is the number of states 
		self.pars = np.linspace(1,6,nmax)
		coh = lambda pos: qt.coherent(Nm,pos,method='analytic')
		self.states = [(coh(Cpos/2) + coh(-Cpos/2)).unit() for Cpos in self.pars]
		self.data_label = 'Fidelity_Cat_Even'
		self.nmax = nmax


class CatsOdd:
	def __init__(self, Nm, nmax = 300): #nmax is the number of states 
		self.pars = np.linspace(1,6,nmax)
		coh = lambda pos: qt.coherent(Nm,pos,method='analytic')
		self.states = [(coh(Cpos/2) - coh(-Cpos/2)).unit() for Cpos in self.pars]
		self.data_label = 'Fidelity_Cat_Odd'
		self.nmax = nmax


class Focks:
	def __init__(self, Nm, nmax = 14): #nmax is the number of states 
		self.pars = range(1,nmax+1)
		self.states = [qt.fock(Nm,n) for n in self.pars]
		self.data_label = 'Fidelity_Fock'
		self.nmax = nmax


class FocksSup:
	def __init__(self, Nm, nmax = 14): #nmax is the number of states 
		self.pars = range(1,nmax+1)
		self.states = [(qt.fock(Nm,0) + qt.fock(Nm,n)).unit() for n in self.pars]
		self.data_label = 'Fidelity_Fock_01'
		self.nmax = nmax


		
