#function to generate a connectivity matrix and a matrix of synaptic time-csts
#for the EIF_network_5+ codes

#takes for argument a dictionary containing the variance of connectivity profiles on a ring
#and the different synaptic time-csts that go with it

import numpy as np
from math import exp
global pi
pi=np.pi


def EIF_architecture(N,E_frac,connectivity_sigmas,connectivity_prob,connectivity_weigths,time_constants):




	#UNPACKING PARAMETERS-----------

	#connectivity profiles standard deviations
	EE_sig=connectivity_sigmas['EE']
	EI_sig=connectivity_sigmas['EI']
	IE_sig=connectivity_sigmas['IE']
	II_sig=connectivity_sigmas['II']

	#connectivity probabilities
	EE_prob=connectivity_prob['EE']
	EI_prob=connectivity_prob['EI']
	IE_prob=connectivity_prob['IE']
	II_prob=connectivity_prob['II']

	#connectivity strengths
	EE_w=connectivity_weigths['EE']
	EI_w=connectivity_weigths['EI']
	IE_w=connectivity_weigths['IE']
	II_w=connectivity_weigths['II']

	#synaptic time-constants for fast synapses (AMPA & GABBA)
	EE_tau=time_constants['EE']
	EI_tau=time_constants['EI']
	IE_tau=time_constants['IE']
	II_tau=time_constants['II']
	#synaptic time-csts fro NMDA synapses
	NMDA_EE_tau=time_constants['NMDA_EE']
	NMDA_EI_tau=time_constants['NMDA_EI']

	#fraction of NMDA syn weigth
	NMDA_vs_AMPA_wFrac_I=connectivity_weigths['I_NMDA_frac']
	NMDA_vs_AMPA_wFrac_E=connectivity_weigths['E_NMDA_frac']

	#ASSIGNING E/I INDICES
	num_I=int(round(N*(1.-E_frac)))
	num_E=N-num_I
	I_index=[int(n*1./(1.-E_frac)) for n in range(num_I)]
	E_index=[n for n in range(N) if n not in I_index ]

	#BULDING CONNECTIVITY---------------------------
	#note: distances are computed over [0,pi] circle
	W=np.zeros([N,N],dtype=float) #connectivity matrix SUPERFLUOUS NOW!!!!!
	Tau=np.zeros([N,N],dtype=float) #time-cst matrix SUPERFLUOUS NOW!!!!!!!
	
	#EE connectivity+++++++++++
	#---
	sig=EE_sig
	num_pre=num_E
	num_post=num_E
	w=EE_w
	tau=EE_tau
	pre_index=E_index
	post_index=E_index
	c_prob=EE_prob
	#---
	for pre in range(num_pre):
		for post in range(num_post):
			dist=circle_dist(pre,post,num_pre,num_post) #'distance' between two cells
			prob=exp(-dist**2/(2.*sig**2))*c_prob
			if np.random.uniform() < prob:
				W[post_index[post],pre_index[pre]]=w
				Tau[post_index[post],pre_index[pre]]=tau 
	#++++++++++++++++++++++++++++

	#EI connectivity+++++++++++
	#---
	sig=EI_sig
	num_pre=num_E
	num_post=num_I
	w=EI_w
	tau=EI_tau
	pre_index=E_index
	post_index=I_index
	c_prob=EI_prob
	#---
	for pre in range(num_pre):
		for post in range(num_post):
			dist=circle_dist(pre,post,num_pre,num_post) #'distance' between two cells
			prob=exp(-dist**2/(2.*sig**2))*c_prob
			if np.random.uniform() < prob:
				W[post_index[post],pre_index[pre]]=w
				Tau[post_index[post],pre_index[pre]]=tau
	#++++++++++++++++++++++++++++

	#IE connectivity+++++++++++
	#---
	sig=IE_sig
	num_pre=num_I
	num_post=num_E
	w=IE_w
	tau=IE_tau
	pre_index=I_index
	post_index=E_index
	c_prob=IE_prob
	#---
	for pre in range(num_pre):
		for post in range(num_post):
			dist=circle_dist(pre,post,num_pre,num_post) #'distance' between two cells
			prob=exp(-dist**2/(2.*sig**2))*c_prob
			if np.random.uniform() < prob:
				W[post_index[post],pre_index[pre]]=w
				Tau[post_index[post],pre_index[pre]]=tau
	#++++++++++++++++++++++++++++

	#II connectivity+++++++++++
	#---
	sig=II_sig
	num_pre=num_I
	num_post=num_I
	w=II_w
	tau=II_tau
	pre_index=I_index
	post_index=I_index
	c_prob=II_prob
	#---
	for pre in range(num_pre):
		for post in range(num_post):
			dist=circle_dist(pre,post,num_pre,num_post) #'distance' between two cells
			prob=exp(-dist**2/(2.*sig**2))*c_prob
			if np.random.uniform() < prob:
				W[post_index[post],pre_index[pre]]=w
				Tau[post_index[post],pre_index[pre]]=tau
	#++++++++++++++++++++++++++++

	#making sure there are no autapses
	for n in range(N): W[n,n]=0.

	#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	#BUILDING CONNECTION INDICES AND WEIGTHS
	id_dict=[0 for n in range(N)] #identity of cell n 0=E, 1=I
	post_indices=[[] for n in range(N)]#indices of cells that cell n projects to
	w_list=[[] for n in range(N)]#postsynaptic weigth for cell n
	tau_array=np.ndarray([N,3],dtype=float) #0 dim: acting on excit ; 1 dim: acting on inhib ; 2:
	NMDA_frac_dict=np.ndarray(N,dtype=float)#fraction of weight on NMDA

	#Running through presyn cells
	for pre in range(N):
		con_ind=[]
		con_w=[]
		con_NMDA_frac=[]
		for post in range(N):
			if W[post,pre]!=0: 
				con_ind.append(post)
				con_w.append(W[post,pre])
		#assigning id
		if pre in E_index:#max(con_w)>0:
			id_dict[pre]=0
			NMDA_frac_dict[pre]=NMDA_vs_AMPA_wFrac_E
		else:
			id_dict[pre]=1
			NMDA_frac_dict[pre]=NMDA_vs_AMPA_wFrac_I
		#storing post-syn indices
		post_indices[pre]=np.array(con_ind)
		w_list[pre]=np.array(con_w)

	#Assigning synaptic time constants for synaptic variables of postsynaptic cells
	for n in range(N): #running over postsynaptic cells
		if id_dict[n]==0: #if cell n is E
			tau_array[n,0]=EE_tau #t-cst E-->E
			tau_array[n,1]=IE_tau #t-cst I-->E
			tau_array[n,2]=NMDA_EE_tau #t-cst NMDA E-->E
		else:
			tau_array[n,0]=EI_tau #t-cst E-->I
			tau_array[n,1]=II_tau #t-cst I-->I
			tau_array[n,2]=NMDA_EI_tau #t-cst NMDA E-->I

	#Generating specific E and I post indices
	E_post_indices=[[] for n in range(N)]
	I_post_indices=[[] for n in range(N)]
	wE_list=[[] for n in range(N)]
	wI_list=[[] for n in range(N)]
	for pre in range(N):
		etemp=[]
		itemp=[]
		wetemp=[]
		witemp=[]
		for post in post_indices[pre]:
			if id_dict[post]==0:
				etemp.append(post)
				wetemp.append(W[post,pre])
			else:
				itemp.append(post)
				witemp.append(W[post,pre])
		E_post_indices[pre]=np.array(etemp)
		I_post_indices[pre]=np.array(itemp)
		wE_list[pre]=np.array(wetemp)
		wI_list[pre]=np.array(witemp)

	return W,Tau,E_index,I_index,id_dict,post_indices,E_post_indices,I_post_indices,w_list,wE_list,wI_list,tau_array,NMDA_frac_dict


#function returning the dstance of two cells on their cricles normalize to their resp E/I numbers
def circle_dist(pre,post,num_pre,num_post):
	pre_position=float(pre)/float(num_pre)
	post_position=float(post)/float(num_post)
	return pi*min(abs(pre_position-post_position),1.-abs(pre_position-post_position))




