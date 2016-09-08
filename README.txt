Guillaume Lajoie January 2015 (http://faculty.washington.edu/glajoie/wordpress/)

DISCLAIMER: This software was design for specialized scientific computation only and is not meant as a deployable product. Some parameter settings may lead to errors.

#####################################################
More details about the network model can be found in:

-Dynamic signal tracking in a simple V1 spiking model, Guillaume Lajoie, Lai-Sang Young, Neural Computations, (2016), Vol. 28, No. 9, Pages 1985-2010, DOI:10.1162/NECO_a_00868
#####################################################

OVERVIEW: 

This model is meant to represent the activity of cortical neurons in visual cortex (area V1) in response to changing stimuli. It enables to explore the effect of different connectivity and dynamics on the neural activity leading to visual perception and image tracking.

Python notebook simulates a network of exponential integrate-and-fire (EIF) neurons arranged in a ring representing their orientation tuning preference. Connectivity between neurons also depends on tuning. There are both excitatory and inhibitory neurons. An external signal representing an oriented edge drives activity in the network.

The script is a simple example using a rotating signal with increasing strength. 


