import numpy as np
import scipy.linalg as lin
import scipy.optimize as scopt
import scipy.integrate as integ

ZeroMat = lambda dim: np.zeros(dim) + 1j*np.zeros(dim)

def eps_corr(theta,A,N_tau,order):

	## Order = 0 means no corrections. Order = 1 means first order corrections. Order = 1.5 means additional second order corrections but only for theta. Order = 2 means all second order corrections.
	if (order not in [0,1,1.5,2]):
		raise ValueError('Order should be either 0, 1, 1.5, or 2')
	
	eps_0 = theta*A/(2*N_tau)

	if (order == 0):
		return np.array([0]),np.array([eps_0])
	
	no_params = 5 if order == 1 else 10
	indices = np.array(range(no_params))
	t_g = N_tau*np.pi/A

	eps_mu_0 = np.zeros(no_params)
	eps_mu_0[0] = eps_0

	def G(n,arg):
		return integ.quad(lambda t: np.cos(n*t - arg*np.sin(t)),0,np.pi/2)[0]

	v_theta = 2*t_g/((2*indices+1)*np.pi)
	v_TA = (2*t_g/np.pi)*(2*indices+1)/(4*N_tau**2-(2*indices+1)**2)

	v_pos = np.array([(-1)**mu*(t_g/np.pi)*(G(N_tau+2*mu+1,theta/2) + G(N_tau-2*mu-1,theta/2)) for mu in indices])

	v_neg = np.array([(-1)**mu*(t_g/np.pi)*(G(N_tau+2*mu+1,-theta/2) + G(N_tau-2*mu-1,-theta/2)) for mu in indices])

	Dyn_1 = np.array([v_theta,v_TA,v_pos,v_neg])
	S_1 = np.array([theta,0,0,0]) - Dyn_1.dot(eps_mu_0)

	sol = scopt.minimize(lambda x: np.linalg.norm(Dyn_1.dot(x) - S_1) + 0.1*np.linalg.norm(x/eps_0),np.zeros(no_params) )
	eps_mu_1 = sol['x']

	eps_mu_corr_1 = eps_mu_0 + eps_mu_1

	if (order == 1):
		return indices,eps_mu_corr_1
	
	## The rest of the function is just for testing higher order corrections and is not used in the main code.
	
	## Removing second order theta-corrections
	
	integ_munu = lambda mu,nu,f: integ.dblquad(lambda u2,u1: np.sin((2*mu+1)*u1)*np.sin((2*nu+1)*u2)*f(u2,u1),0,np.pi,lambda u1: 0, lambda u1:u1)[0]  # If it is too slow, then tolerances can be changed, e.g. epsabs = -1, epsrel = 1e-3

	SC_fun = lambda u2,u1: np.sin(N_tau*(u1-u2))*np.cos((np.cos(u1)-np.cos(u2))*theta/2)
	CS_fun = lambda u2,u1: np.cos(N_tau*(u1-u2))*np.sin((np.cos(u1)-np.cos(u2))*theta/2)

	def Fun_to_Mat(fun):
		Ifun = np.zeros([no_params,no_params])
		for i in range(no_params):
			mu = indices[i]
			Ifun[i,i] = integ_munu(mu,mu,fun)
			for j in range(no_params):
				nu = indices[j]
				if (nu<mu):
					Ifun[i,j] = integ_munu(mu,nu,fun)
					Ifun[j,i] = Ifun[i,j]
		return Ifun
			
	ISC = (t_g/np.pi)**2 * Fun_to_Mat(SC_fun)
	ICS = (t_g/np.pi)**2 * Fun_to_Mat(CS_fun)

	eps_guess = eps_mu_corr_1
	Orth_proj = lin.null_space([v_TA,v_pos,v_neg])  #Projecting onto this subspace reduces the computational time
	
	def Mat_to_func(I):
		c = eps_guess.dot(I).dot(eps_guess)
		l = 2*eps_guess.dot(I).dot(Orth_proj)
		q = np.transpose(Orth_proj).dot(I).dot(Orth_proj)
		return lambda eps_p: c + l.dot(eps_p) + (q.dot(eps_p)).dot(eps_p)
	
	Qfun_SC = Mat_to_func(ISC)
	Qfun_CS_sec = Mat_to_func(ICS)
	Qfun_CS = lambda eps_p: Qfun_CS_sec(eps_p) - v_theta.dot(eps_guess - eps_mu_0) - v_theta.dot(Orth_proj).dot(eps_p)

	if (order == 1.5):
		sol = scopt.minimize(lambda eps_p: Qfun_SC(eps_p)**2 + Qfun_CS(eps_p)**2 ,np.zeros(len(Orth_proj[0])))  
		eps_perp = sol['x']

		eps_mu_corr_1_5 = eps_guess + Orth_proj.dot(eps_perp)

		return indices,eps_mu_corr_1_5

	## Removing other second order corrections

	#SE_fun = lambda u2,u1: np.sin(N_tau*(u1-u2)) * np.cos((np.cos(u1)+np.cos(u2))*theta/2)
	#ISE = (t_g/np.pi)**2 * Fun_to_Mat(SE_fun)
	
	sign_mat = np.array([[(-1)**(indices[i]+indices[j]) for i in range(no_params)] for j in range(no_params)])
	Term1_fun = lambda arg: np.array([[G(3*N_tau-2*nu+2*mu,arg)*(1/(2*N_tau-2*nu-1) + 1/(2*N_tau+2*mu+1)) for mu in indices] for nu in indices])
	Term2_fun = lambda arg: np.array([[G(3*N_tau+2*nu-2*mu,arg)*(1/(2*N_tau+2*nu+1) + 1/(2*N_tau-2*mu-1)) for mu in indices] for nu in indices])
	Term3_fun = lambda arg: np.array([[G(3*N_tau-2*nu-2*mu-2,arg)*(1/(2*N_tau-2*nu-1) + 1/(2*N_tau-2*mu-1)) for mu in indices] for nu in indices])
	Term4_fun = lambda arg: np.array([[G(3*N_tau+2*nu+2*mu+2,arg)*(1/(2*N_tau+2*nu+1) + 1/(2*N_tau+2*mu+1)) for mu in indices] for nu in indices])
	Terms_fun = lambda arg: Term1_fun(arg) + Term2_fun(arg) + Term3_fun(arg) + Term4_fun(arg)

	I03 = (t_g/np.pi)**2 * np.sqrt(3)/2 * sign_mat*Terms_fun(theta/2)
	I13 = (t_g/np.pi)**2 * np.sqrt(3)/2 * sign_mat*Terms_fun(-theta/2)

	def G_quad_fact(x):	
		return (1/2)*(t_g/np.pi)**2 * np.array([[
			(-1)**(mu+nu) * (1/(2*mu+1)) * ( G(N_tau - 2*(nu-mu),x) - G(N_tau + 2*(nu-mu),x) + G(N_tau+2*(mu+nu+1),x) - G(N_tau-2*(mu+nu+1),x) ) 
			for mu in indices] for nu in indices])

	G_quad_pos = (1/2)*(t_g/np.pi)**2 * G_quad_fact(theta/2)
	G_quad_neg = (1/2)*(t_g/np.pi)**2 * G_quad_fact(-theta/2)

	eps_guess = eps_mu_corr_1
	Orth_proj = lin.null_space([v_TA,v_pos,v_neg])  #Projecting onto this subspace reduces the computational time

	c_pos = eps_guess.dot(G_quad_pos).dot(eps_guess - eps_mu_0)
	l_pos = eps_guess.dot(G_quad_pos).dot(Orth_proj) + (eps_guess - eps_mu_0).dot(np.transpose(G_quad_pos)).dot(Orth_proj)
	q_pos = np.transpose(Orth_proj).dot(G_quad_pos).dot(Orth_proj)
	Qfun_Gpos = lambda eps_p: c_pos + l_pos.dot(eps_p) + q_pos.dot(eps_p).dot(eps_p)

	c_neg = eps_guess.dot(G_quad_neg).dot(eps_guess - eps_mu_0)
	l_neg = eps_guess.dot(G_quad_neg).dot(Orth_proj) + (eps_guess - eps_mu_0).dot(np.transpose(G_quad_neg)).dot(Orth_proj)
	q_neg = np.transpose(Orth_proj).dot(G_quad_neg).dot(Orth_proj)
	Qfun_Gneg = lambda eps_p: c_neg + l_neg.dot(eps_p) + q_neg.dot(eps_p).dot(eps_p)

	#Qfun_SE = Mat_to_func(ISE)
	Qfun_03 = Mat_to_func(I03)
	Qfun_13 = Mat_to_func(I13)

	sol = scopt.minimize(lambda eps_p: Qfun_SC(eps_p)**2 + Qfun_CS(eps_p)**2 + Qfun_Gpos(eps_p)**2 + Qfun_Gneg(eps_p)**2 + Qfun_03(eps_p)**2 + Qfun_13(eps_p)**2 ,np.zeros(len(Orth_proj[0])))  # Somehow Qfun_SE is inconsistent with others, don't know why
	eps_perp = sol['x']

	eps_mu = eps_guess + Orth_proj.dot(eps_perp)

	return indices,eps_mu

def timeexp(func,t,freq_scale):
	res = lin.expm(ZeroMat(func(1).shape))
	dt = 0.05/freq_scale
	t_nums = int(round(t/dt))
	t_lin = np.linspace(0,t,t_nums)
	for t_step in t_lin:
		res = lin.expm(1j*func(t_step+dt/2)*dt).dot(res)
	return res


