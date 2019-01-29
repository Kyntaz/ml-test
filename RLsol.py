# -*- coding: utf-8 -*-
# Pedro Quintas, 83546
# Gonçalo Gaspar, 83471
# Grupo 95

"""
Created on Mon Oct 16 20:31:54 2017

@author: mlopes
"""
import numpy as np

def Q2pol(Q, eta=5):
	# Use an exponential function in order to dramatize the probabilities.
	pol = np.zeros((len(Q), len(Q[0])))
	aQ = eta * Q
	base = 3
	for i in range(len(Q)):
		l_tot = sum([base**a for a in aQ[i]])
		for j in range(len(Q[i])):
			pol[i][j] = base**aQ[i][j] / l_tot

	#print(pol)
	return pol

class myRL:

	def __init__(self, nS, nA, gamma):
		self.nS = nS
		self.nA = nA
		self.gamma = gamma
		self.Q = np.zeros((nS,nA))
		
	def traces2Q(self, trace):
		ts = 5
		for i in range(ts):
			alpha = 1 - (i * (1/ts))
			for traj in trace:
				a = int(traj[1])
				s = int(traj[0])
				s1 = int(traj[2])
				r = int(traj[3])

				# Update Qs
				self.Q[s][a] = (1-alpha) * self.Q[s][a] + alpha * (r + self.gamma * self.Q[s1].max())
		
		return self.Q




			