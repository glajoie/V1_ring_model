#GL 2015
#Script to simulate a a network of EIF neurons 

#Details can be found in:
#---
#Dynamic signal tracking in a simple V1 spiking model, Guillaume Lajoie,
# Lai-Sang Young, Neural Computations, (2016), 
#Vol. 28, No. 9, Pages 1985-2010, DOI:10.1162/NECO_a_00868
#---

import numpy as np
from math import exp, sin, pi, sqrt
import scipy.io as io
from sys import stdout
import time as tim

#-------------------------------
#SOLVER MAIN FUNCTION
#-------------------------------
def integrate_EIF(p):
	#UNPACKING PARAMETER DICTIONARIES+++++

	#switches deciding what to save
	traj_switch=p['traj_switch'],
	synaptic_switch=p['synaptic_switch'],
	spike_switch=p['spike_switch'],
	input_switch=p['input_switch'],

	#solver
	dt=p['dt']
	t_span=p['t_span']
	t_record=p['t_record'] #save traj and/or inputs every t_record
	IC=p['IC'] #vector of initial condition
	N_rec_index=p['N_rec_index'] #index of cells to record from

	#model neuron params
	global C_inv,gL,DT,VT,EL,IT
	C_inv=1./p['C']
	gL=p['gL']
	DT=p['DT']
	VT=p['VT']
	EL=p['EL']
	IT=p['IT']
	tref=p['tref'] #refractory period
	Vth=p['Vth'] #Threshold voltage
	Vr=p['Vr'] #Reset voltage

	#network parameters
	global N, num_E, num_I, E_index, I_index
	N=p['N']
	# W=p['W']
	# Tau=p['Tau']#synaptic decay t-csts

	#synaptic input params
	E_index=p['E_index']#index of E cells
	I_index=p['I_index']#index of I_cells
	num_E=len(E_index)
	num_I=N-num_E
	EE_delay=p['EE_delay']#synaptic delay
	IE_delay=p['IE_delay']
	EI_delay=p['EI_delay']
	II_delay=p['II_delay']
	id_dict=p['id_dict']
	# post_indices=p['post_indices']
	E_post_indices=p['E_post_indices']
	I_post_indices=p['I_post_indices']
	# w_list=p['w_list']
	wE_list=p['wE_list']
	wI_list=p['wI_list']
	tau_array=p['tau_array']
	NMDA_frac_dict=p['NMDA_frac_dict']

	#OTHER INPUT PARAMS
	ext_w=p['ext_w'] #weigths forexternal inputs (orientation and background)
	I=p['I'] #constant offset
	eps=p['eps']#variance of white noise

	#orientation input params
	or_switch_times=p['or_switch_times'] #vector
	or_values=p['or_values'] #vector (same size as above)
	or_tuning=p['or_tuning'] #scalar
	or_strengths=p['or_strengths']	#vector

	#+++++++++++++++++++++++++++++++++++++++

	#MISC code initialization+++++++++++
	spike_index=np.array([]) #to track which neurons spike at each time step [!!!BOTTLENECK!!! as resized always]
	spike_count=np.zeros(N)
	print_prog=0. #keep trak of time units to display as code runs
	N_rec=len(N_rec_index) #number of neurons to record from
	if traj_switch or synaptic_switch: #for recording purpuses
		next_t_rec=t_record 
	else:
		next_t_rec=2.*t_span

	#TRAJECTORY MANAGEMENT
	x_new=np.zeros(N,dtype=float) #iteration vectors
	x_now=np.zeros(N,dtype=float)
	ref_t=-dt*np.ones(N,dtype=float)#container for end of refractory periods
	In_val=np.zeros(N,dtype=float)

	#SYNAPTIC MANAGEMENT
	s_now=np.zeros([N,3],dtype=float) #synaptic variable vectors 0:E and 1:I and 2:NMDA
	s_new=np.zeros([N,3],dtype=float)
	#cue stuff
	cue_len=int(max(EE_delay,EI_delay,IE_delay,II_delay)/dt*max(num_E,num_I))+5
	spike_cue=(t_span+10)*np.ones([cue_len,4]) #second dim is for EE:0, EI:1, IE:2, II:3
	index_cue=[[0 for n in range(cue_len)],[0 for n in range(cue_len)],[0 for n in range(cue_len)],[0 for n in range(cue_len)]] #first dim is for id E:0 I:1
	cue_entry=[0,0,0,0] #circular cue index (dim is for id EE:0 EI:1 IE:2 II:3)
	cue_exit=[0,0,0,0] #circular cue index (dim is for id EE:0 EI:1 IE:2 II:3)
	


	#DATA CONTAINERS INITIALIZATION+++++
	rec_length=int(np.floor(t_span/t_record)-1) #size of continuous-time containers (add 1 to save at t=0)
	if traj_switch or synaptic_switch: 
		time=np.ndarray(rec_length,dtype=float) #saved times array
	if traj_switch: 
		traj=np.zeros([N_rec,rec_length],dtype=float) #traj array
	if synaptic_switch: 
		syn_input=np.zeros([N_rec,3,rec_length],dtype=float)
	if input_switch:
		ori_input=np.zeros([N_rec,rec_length],dtype=float)
		noise_input=np.zeros([N_rec,rec_length],dtype=float)
	if spike_switch: 
		spike_array=np.zeros([N_rec,np.ceil(1.5*t_span)]) #make it big enough so it doe not overflow
		spike_ar_index=[0 for n in range(N_rec)]
		# spikes=[[] for n in range(N_rec)]#spikes container [!!!BOTTLENECK!!! is growing without being initialized]

	#ORIENTATION SIGNAL VARIABLES initilazation
	signal_index=0
	next_or_switch=or_switch_times[signal_index]
	I_or=or_strengths[signal_index]*orientation_signal(or_values[signal_index],or_tuning) #initializing signal
	

	#INTEGRATION++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	# Start timer:
	tBegin = tim.time() #for time estimates
	print('Integrating network dynamics...')
	x_now=IC
	rec_index=0
	for i in range(int(np.floor(t_span/dt))):
		t_now=i*dt

		
		#--------------------------------------------------
		#COMPUTING TOTAL INPUT VALUE at t_now
		#--------------------------------------------------
		#Checking for orientation signal switch and change value of I_or
		if t_now>next_or_switch:
			if signal_index<len(or_switch_times)-1:
				signal_index+=1
				next_or_switch=or_switch_times[signal_index]
				I_or=or_strengths[signal_index]*orientation_signal(or_values[signal_index],or_tuning)
				I_or=ext_w*I_or #scaling
		#computing background input (cst I + noise)
		I_noise=I+eps*np.random.normal(0.,1.,N)*1./sqrt(dt)
		I_noise=ext_w*I_noise #scaling

		#total synaptic input vector
		In_val=np.sum(s_now,1)+I_noise+I_or
		#--------------------------------------------------

		#--------------------------------------------------
		#SAVING RECORDS AT t_now
		#--------------------------------------------------
		#saving input for this t-step
		if t_now>=next_t_rec:
			if traj_switch or synaptic_switch:
				time[rec_index]=t_now
			if traj_switch:
				traj[:,rec_index]=x_now[N_rec_index]
			if synaptic_switch:
				syn_input[:,:,rec_index]=s_now[N_rec_index,:]#In_val[N_rec_index]-I_or[N_rec_index]
				# noise_input[:,rec_index]=I_noise[N_rec_index]
			if input_switch:
				ori_input[:,rec_index]=I_or[N_rec_index]
				noise_input[:,rec_index]=I_noise[N_rec_index]
			next_t_rec+=t_record
			rec_index+=1
		#--------------------------------------------------

		#--------------------------------------------------
		#INTEGRATING for t_now+dt (and inforcing refractory period)
		#--------------------------------------------------
		#SOLVER
		#Euler
		x_new=x_now+dt*EIF_VF(t_now,x_now,In_val)
		ref_index=np.where(ref_t>=t_now)[0] #finding indices where refractory period is over
		x_new[ref_index]=Vr #leaving neurons in refractory period at Vr

		#--------------------------------------------------
		##detecting a spike--------------------------------
		#-------------------------------------------------- 
		#[!!!BOTTLENECK!!!](?)
		spike_index=np.where(x_new>Vth)[0] #indices of neurons over threshold at i+1 Vth
		spike_count[spike_index]+=1
		if spike_index.size>0:
			ref_t[spike_index]=t_now+dt/2.+tref #updating refractory times
			x_new[spike_index]=Vr #reset cells that would be over threshold at i+1
			#------------------------------------------
			#CUES UPDATING
			#------------------------------------------
			for n in spike_index: 
				current_id=id_dict[n] #extracting id of cell
				if current_id==0: #E cell spiking
					#Update EE cue
					spike_cue[cue_entry[0],0]=t_now+dt/2.+EE_delay
					#updating corresponding spiking index cue
					index_cue[0][cue_entry[0]]=n
					#updating cue_entry
					cue_entry[0]=(cue_entry[0]+1)%cue_len

					#Update EI cue
					spike_cue[cue_entry[1],1]=t_now+dt/2.+EI_delay
					#updating corresponding spiking index cue
					index_cue[1][cue_entry[1]]=n
					#updating cue_entry
					cue_entry[1]=(cue_entry[1]+1)%cue_len
				else: #I cell spiking
					#Update IE cue
					spike_cue[cue_entry[2],2]=t_now+dt/2.+IE_delay
					#updating corresponding spiking index cue
					index_cue[2][cue_entry[2]]=n
					#updating cue_entry
					cue_entry[2]=(cue_entry[2]+1)%cue_len

					#Update II cue
					spike_cue[cue_entry[3],3]=t_now+dt/2.+II_delay
					#updating corresponding spiking index cue
					index_cue[3][cue_entry[3]]=n
					#updating cue_entry
					cue_entry[3]=(cue_entry[3]+1)%cue_len
			#------------------------------------------
				
				#------------------------------------------
				#SAVING SPIKES
				#------------------------------------------
				if spike_switch and n in N_rec_index: #SAVE spikes
					spike_array[N_rec_index.index(n),spike_ar_index[N_rec_index.index(n)]]=t_now+dt/2.
					spike_ar_index[N_rec_index.index(n)]+=1 
		#--------------------------------------------------

		#--------------------------------------------------
		#INTEGRATING SYNAPTIC VARIABLES for t_now+dt
		#--------------------------------------------------
		s_new=s_now-dt*s_now/tau_array #updating both E and I synaptic variable at each cells
		
		# checking EE spike cue
		while t_now>=spike_cue[cue_exit[0],0]-dt and cue_exit[0]!=cue_entry[0]: #checking if spike will occur in next tim step
			#retrieving index of spiking cell !!!!!!!
			current_index=index_cue[0][cue_exit[0]]
			#retrieving post-syn connected cells !!!!!!
			current_post_indices=E_post_indices[current_index]
			#retrieving current outgoing weights !!!!!!
			current_ws=wE_list[current_index]
			#current NMDA fraction
			current_NMDA_frac=NMDA_frac_dict[current_post_indices]
			#making translations for fast E synapses
			s_new[current_post_indices,0]=s_now[current_post_indices,0]+current_ws/tau_array[current_post_indices,0]*(1.-current_NMDA_frac)
			#making translations for slow NMDA synapses
			s_new[current_post_indices,2]=s_now[current_post_indices,2]+current_ws/tau_array[current_post_indices,2]*current_NMDA_frac
			#updating exit cue !!!!!
			cue_exit[0]=(cue_exit[0]+1)%cue_len

		# checking EI spike cue
		while t_now>=spike_cue[cue_exit[1],1]-dt and cue_exit[1]!=cue_entry[1]: #checking if spike will occur in next tim step
			#retrieving index of spiking cell !!!!!!!
			current_index=index_cue[1][cue_exit[1]]
			#retrieving post-syn connected cells !!!!!!
			current_post_indices=I_post_indices[current_index]
			#retrieving current outgoing weights !!!!!!
			current_ws=wI_list[current_index]
			#current NMDA fraction
			current_NMDA_frac=NMDA_frac_dict[current_post_indices]
			#making translations for fast E synapses
			s_new[current_post_indices,0]=s_now[current_post_indices,0]+current_ws/tau_array[current_post_indices,0]*(1.-current_NMDA_frac)
			#making translations for slow NMDA synapses
			s_new[current_post_indices,2]=s_now[current_post_indices,2]+current_ws/tau_array[current_post_indices,2]*current_NMDA_frac
			#updating exit cue !!!!!
			cue_exit[1]=(cue_exit[1]+1)%cue_len

		# checking IE spike cue
		while t_now>=spike_cue[cue_exit[2],2]-dt and cue_exit[2]!=cue_entry[2]: #checking if spike will occur in next tim step
			#retrieving index of spiking cell
			current_index=index_cue[2][cue_exit[2]]
			#retrieving post-syn connected cells
			current_post_indices=E_post_indices[current_index]
			#retrieving current outgoing weights
			current_ws=wE_list[current_index]
			#making translations
			s_new[current_post_indices,1]=s_now[current_post_indices,1]+current_ws/tau_array[current_post_indices,1]
			#updating exit cue
			cue_exit[2]=(cue_exit[2]+1)%cue_len

		# checking II spike cue
		while t_now>=spike_cue[cue_exit[3],3]-dt and cue_exit[3]!=cue_entry[3]: #checking if spike will occur in next tim step
			#retrieving index of spiking cell
			current_index=index_cue[3][cue_exit[3]]
			#retrieving post-syn connected cells
			current_post_indices=I_post_indices[current_index]
			#retrieving current outgoing weights
			current_ws=wI_list[current_index]
			#making translations
			s_new[current_post_indices,1]=s_now[current_post_indices,1]+current_ws/tau_array[current_post_indices,1]
			#updating exit cue
			cue_exit[3]=(cue_exit[3]+1)%cue_len
		#--------------------------------------------------


		#updating
		x_now=x_new
		s_now=s_new
		#--------------------------------------------------
		#printing progress
		if np.floor(100.*(t_now/t_span))>=print_prog:
			tEnd = tim.time()
			t_lapse=tEnd-tBegin
			prog=t_now/t_span
			if prog>0:
				est_T=t_lapse/prog
			else:
				est_T=100.
			if (est_T-t_lapse)>60.:
				mins_remaining='about '+str(np.floor((est_T-t_lapse)/60.))+' minutes remaining'
			else:
				mins_remaining='about '+str(np.floor(est_T-t_lapse))+' seconds remaining'
			print_prog=np.floor(100.*(t_now/t_span))

			stdout.write('                                                   ')
			stdout.write('\r')
			# stdout.flush()
			stdout.write('Simulation progress: '+str(int(print_prog))+' %, '+mins_remaining)
			# stdout.write('\r')
			# stdout.flush()
		#--------------------------------------------------

	#PACKAGING TO RETURN
	out={'spike_count':spike_count}
	if traj_switch or synaptic_switch:
		out['time']=time
	if traj_switch:
		out['traj']=traj
	if synaptic_switch:
		out['syn_input']=syn_input
	if spike_switch:
		#dismount spike_array into list container
		spikes=[spike_array[n,:spike_ar_index[n]] for n in range(N_rec)]
		out['spikes']=spikes
	if input_switch:
		out['ori_input']=ori_input
		out['noise_input']=noise_input

	return out 


#-------------------------------
#OTHER FUNCTIONS
#-------------------------------
#-------------------------------
#VECTOR FIELD
#-------------------------------
def EIF_VF(t,V,Iin):
	return C_inv*(-gL*(V-EL)+gL*DT*np.exp((V-VT)/DT)+Iin)

def orientation_signal(orientation_value,tuning): #returns a vector of current values (highest one is 1.)
	#orientation is between 0 and pi
	#tuning is standard dev
	#This below returns the orientation signal to all cells

	#OLD
	# x=np.array(range(N))*pi/N
	# dist=np.array([min((y-orientation_value)%pi,(orientation_value-y)%pi) for y in x])
	# return np.exp(-dist**2./(2.*tuning**2))

	#NEW
	z=np.zeros(N,dtype=float) #container for output
	#E cells
	x=np.array(range(num_E))*pi/num_E
	dist=np.array([min((y-orientation_value)%pi,(orientation_value-y)%pi) for y in x])
	z[E_index]=np.exp(-dist**2./(2.*tuning**2))
	#I cells
	x=np.array(range(num_I))*pi/num_I
	dist=np.array([min((y-orientation_value)%pi,(orientation_value-y)%pi) for y in x])
	z[I_index]=np.exp(-dist**2./(2.*tuning**2))
	return z

#---------------------------------------------------
#---------------------------------------------------
#---------------------------------------------------




