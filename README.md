# ASTR496-Assignment-3
Assignment 3 for ASTR 496

## Explanation of What's Included


### Assignment_3_Write_Up
This is a jupyter notebook with all of the code I used for the later tests I ran. It includes my write up for this assignment. It contains the answers to the questions from the assignment page as well as some discussion about the process I went through for this assignment.

### Assignment_3_Single
This python file executes one run of my evolve function and displays plots(it does not save the plots). All of the inputs for the evolve function must be read in from command line. The inputs are as follows:
* p0: Initial Density (g / cm^3)
* f_H: Initial Hydrogen Ionization Fraction
* f_He: Initial Helium Ionization Fraction
* final_t: The final time to integrate to (sec)
* integrator_type: which integrator in scipy.ode you would like to use
* safety_factor: Used to help define dt. Since I couldn't get the dt to change by the equation SF * min(S/Sdot) working dt is defined as final_t / SF so some large value must be used.
* element_case: 1 = H, Hp, de while 2 = H, Hp, He, Hep, Hepp, de. If I did not have a coupon, I would have made 3 = nine species network
* T0: An initial guess for the temperature that is used to get an initial guess for the energy(which is held constant throughout the simulation)

### Assignment_3_Multiple
This python file runs a series of evolve functions meant to test different combinations of parameters. It tested both element cases for f_H: 0.0, 1e-6, and 1.0 and T0: 10^2 - 10^6. The rest of the inputs were held at constant values. The results from these runs are saved in the Results folder.

### Results
Holds the plots generated from the Assignment_3_Multiple.py file.

### Assignment_3_old
This has a good amount of my old code in it. This was mainly left in just in case you wanted to see some of the things I was trying out in the past. For example, I wanted to originally make my code all work symbolically, but that ran too slow. It also has a few of my attempts to get u and T working correctly in case you wanted to see how overly complicated I was making things.
