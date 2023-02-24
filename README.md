# NNBK

This is my master's thesis project that I'm currently working on. NNBK is a ripoff name from the NNPDF collaboration since they have similar problem with parton distribution functions (PDFs). 

I'm rewriting the program to a more "public friendly" form and try to document it as I go as best as I can. The program doesn't work at the moment, but I will try to get it to a point where it runs ASAP.

## Goal
The goal of this project is to explore if BK equation initial condition could be predicted from $F_2$ structure function values using a neural network. 
The main idea is to train the network with $F_2$ values calculated from several different parametrisations and functional forms for the initial condition and a range of different parameter values so that the network learns the relation between F2 values and BK equation initial condition. The network takes the dipole size r and any number of $F_2(Q^2,x)$ values as an input and predicts the corresponding dipole amplitude $N(r)$.


## Motivation
The motivation is to find a less biased way to define the initial condition. At the moment we don't know what the functional form of the initial condition is so the current method is to assume some parametrisation and fit the parameters to experimental data. This assumption of brings a bias to the calculations and with neural network this bias could be reduced.
