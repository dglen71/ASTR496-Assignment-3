
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets
import sympy
import scipy.integrate as sint
import sys

print(sys.argv)


def k1_function(T):
    T_eV = T / 11605.0
    log_T_eV = np.log(T_eV)
    rv = np.exp(-32.71396786375
          + 13.53655609057*log_T_eV
          - 5.739328757388*log_T_eV**2 
          + 1.563154982022*log_T_eV**3
          - 0.2877056004391*log_T_eV**4
          + 0.03482559773736999*log_T_eV**5
          - 0.00263197617559*log_T_eV**6
          + 0.0001119543953861*log_T_eV**7
          - 2.039149852002e-6*log_T_eV**8)
    return rv

def k2_function(T):
    rv = 4.881357e-6*T**(-1.5)* (1.+1.14813e2 * T**(-0.407))**(-2.242)
    return rv

def k3_function(T):
    T_eV = T / 11605.0
    log_T_eV = np.log(T_eV)
    rv = np.exp(-44.09864886561001
             + 23.91596563469*log_T_eV
             - 10.75323019821*log_T_eV**2
             + 3.058038757198*log_T_eV**3
             - 0.5685118909884001*log_T_eV**4
             + 0.06795391233790001*log_T_eV**5
             - 0.005009056101857001*log_T_eV**6
             + 0.0002067236157507*log_T_eV**7
             - 3.649161410833e-6*log_T_eV**8)
    return rv

def k4_function(T):
    T_eV = T / 11605.0
    rv = (1.54e-9*(1.0+0.3 / 
             np.exp(8.099328789667/T_eV))
             / (np.exp(40.49664394833662/T_eV)*T_eV**1.5)
             + 3.92e-13/T_eV**0.6353)
    return rv

def k5_function(T):
    T_eV = T / 11605.0
    log_T_eV = np.log(T_eV)
    rv = np.exp(-68.71040990212001
             + 43.93347632635*log_T_eV
             - 18.48066993568*log_T_eV**2
             + 4.701626486759002*log_T_eV**3
             - 0.7692466334492*log_T_eV**4
             + 0.08113042097303*log_T_eV**5
             - 0.005324020628287001*log_T_eV**6
             + 0.0001975705312221*log_T_eV**7
             - 3.165581065665e-6*log_T_eV**8)
    return rv

def k6_function(T):
    rv = 7.8155e-5*T**(-1.5) * (1.+2.0189e2* T**(-0.407))**(-2.242)
    return rv 

def k7_function(T):
    T_eV = T / 11605.0
    log_T_eV = np.log(T_eV)
    rv = np.exp(-20.37260896533324
             + 1.139449335841631*log_T_eV
             - 0.1421013521554148*log_T_eV**2
             + 0.00846445538663*log_T_eV**3
             - 0.0014327641212992*log_T_eV**4
             + 0.0002012250284791*log_T_eV**5
             + 0.0000866396324309*log_T_eV**6
             - 0.00002585009680264*log_T_eV**7
             + 2.4555011970392e-6*log_T_eV**8
             - 8.06838246118e-8*log_T_eV**9) 
    return rv

def k8_function(T):
    rv = 1.75e-17 *T**1.3 * np.exp(-1.578 / T)
    return rv

#9,10,11 Could not find in Source Code


def k12_function(T):
    rv = 3.0e-16 * (T/3e2)**0.95 * np.exp(-T/9.32e3)
    return rv

def k13_function(T):
    rv = 1.35e-9*(T**9.8493e-2 + 3.2852e-1 * T**5.5610e-1 + 2.771e-7 * T**2.1826)             / (1 + 6.191e-3 * T**1.0461 + 8.9712e-11 * T**3.0424 + 3.2576e-14 * T**3.7741)
    return rv

def k14_function(T): 
    rv = 2.10e-20 * (T/30.0)**(-0.15) 
    return rv

def k15_function(T = None):
    rv = 6.0e-10
    return rv

def k16_function(T):
    log_T = np.log10(T)
    rv = (np.exp(-21237.15/T) * (- 3.3232183e-07 + 3.3735382e-07 * log_T - 1.4491368e-07 * log_T**2                              + 3.4172805e-08 * log_T**3 - 4.7813720e-09 * log_T**4 + 3.9731542e-10 * log_T**5                             - 1.8171411e-11 * log_T**6 + 3.5311932e-13 * log_T**7))
    return rv

def k17_function(T):
    rv = 4.4886e-9*T**0.109127*np.exp(-101858/T)
    return rv

def k18_function(T):
    T_eV = T/11605
    rv = 1.0670825e-10*T_eV**2.012/(np.exp(4.463/T_eV)*(1+0.2472* T_eV)**3.512)
    return rv


def k19_function(T):
    T_eV = T / 11605.0
    log_T_eV = np.log(T_eV)
    rv = np.exp(-18.01849334273
             + 2.360852208681*log_T_eV
             - 0.2827443061704*log_T_eV**2
             + 0.01623316639567*log_T_eV**3
             - 0.03365012031362999*log_T_eV**4
             + 0.01178329782711*log_T_eV**5
             - 0.001656194699504*log_T_eV**6
             + 0.0001068275202678*log_T_eV**7
             - 2.631285809207e-6*log_T_eV**8)
    return rv

def k20_function(T):
    T_eV = T / 11605.0
    log_T_eV = np.log(T_eV)
    rv = np.exp(-20.37260896533324
             + 1.139449335841631*log_T_eV
             - 0.1421013521554148*log_T_eV**2
             + 0.00846445538663*log_T_eV**3
             - 0.0014327641212992*log_T_eV**4
             + 0.0002012250284791*log_T_eV**5
             + 0.0000866396324309*log_T_eV**6
             - 0.00002585009680264*log_T_eV**7
             + 2.4555011970392e-6*log_T_eV**8
             - 8.06838246118e-8*log_T_eV**9)
    return rv
    
def k21_function(T):
    rv = 2.4e-6*(1.0+T/2e4)/np.sqrt(T)
    return rv
    
def k22_function(T):
    rv = 4.0e-4*T**(-1.4)*np.exp(-15100/T)
    return rv

def k23_function(T):
    rv = 1.32e-6 * T**(-0.76)
    return rv

def k24_function(T):
    rv = 5.e-7*np.sqrt(100.0/T)
    return rv
    
def k25_function(T):
    T_eV = T / 11605.0
    rv = 1.0670825e-10*T_eV**2.012 / (np.exp(4.463/T_eV)*(1.0+0.2472* T_eV)**3.512)
    return rv

#This may not be right?
def k26_function(T):
    rv = 2.8e-31 * (T**(-0.6e0))
    return rv

#27-32 Could not find in the source code


'''
Ts = np.logspace(1, 8, 1024)

plt.loglog(Ts, k1_function(Ts), label = 'k1')
plt.loglog(Ts, k2_function(Ts), label = 'k2')
plt.loglog(Ts, k3_function(Ts), label = 'k3')
plt.loglog(Ts, k4_function(Ts), label = 'k4')
plt.loglog(Ts, k5_function(Ts), label = 'k5')
plt.loglog(Ts, k6_function(Ts), label = 'k6')
plt.loglog(Ts, k7_function(Ts), label = 'k7')
plt.loglog(Ts, k8_function(Ts), label = 'k8')
    
plt.title("Reaction Rates Versus Temperature")
plt.xlabel("Temperature")
plt.ylabel("Reaction Rate")
plt.ylim(1e-20, 1e-6)
plt.legend()
plt.show()

'''

H, Hp, Hm, He, Hep, Hepp, H2, H2p, grain, de = sympy.sympify("H, Hp, Hm, He, Hep, Hepp, H2, H2p, grain, de")
k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15,k16,k17,k18,k19,k20,k21,k22,k23,k24,k25,k26,k27,k28,k29,k30,k31,k32 = sympy.sympify("k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15,k16,k17,k18,k19,k20,k21,k22,k23,k24,k25,k26,k27,k28,k29,k30,k31,k32")


r1 = (H + de), (Hp + de + de), k1
r2 = (Hp + de), (H), k2
r3 = (He + de), (Hep + de + de), k3
r4 = (Hep + de), (He), k4
r5 = (Hep + de), (Hepp + de + de), k5
r6 = (Hepp + de), (Hep), k6
r7 = (H + H), (Hp + de + H), k7
r8 = (H + He), (Hp + de + He), k8
r9 = (H), (Hp + de), k9
r10 = (He), (Hep + de), k10
r11 = (Hep), (Hepp + de), k11
r12 = (H + de), (Hm), k12
r13 = (Hm + H), (H2 + de), k13
r14 = (H + Hp), (H2p), k14
r15 = (H2p + H), (H2 + Hp), k15
r16 = (H2 + Hp), (H2p + H), k16
r17 = (H2 + de), (H + H + de), k17
r18 = (H2 + H), (H + H + H), k18
r19 = (Hm + de), (H + de + de), k19
r20 = (Hm + H), (H + de + H), k20
r21 = (Hm + Hp), (H + H), k21
r22 = (Hm + Hp), (H2p + de), k22
r23 = (H2p + de), (H + H), k23
r24 = (H2p + Hm), (H2 + H), k24
r25 = (H + H + H), (H2 + H), k25
r26 = (H + H + H2), (H2 + H2), k26
r27 = (Hm), (H + de), k27
r28 = (H2p), (H + Hp), k28
r29 = (H2), (H2p + de), k29
r30 = (H2p), (Hp + Hp + de), k30
r31 = (H2), (H + H), k31
r32 = (H + H + grain), (H2 + grain), k32



all_reactions = [r1, r2, r3, r4, r5, r6, r7, r8]


def find_formation(species):
    rxns = []
    for r in all_reactions:
        if species in r[1].atoms():
            rxns.append(r)
    return rxns
    
def find_destruction(species):
    rxns = []
    for r in all_reactions:
        if species in r[0].atoms():
            rxns.append(r)
    return rxns


def get_rhs(species):
    dSdt = 0
    for lhs, rhs, coeff in find_formation(species):
        term = coeff
        for atom in list(lhs.args):
            term *= atom
        for atom in list(rhs.args):
            if ("2*" + str(species) in str(atom)):
                term *= 2
            if ("3*" + str(species) in str(atom)):
                term *= 3
        dSdt += term
    
    for lhs, rhs, coeff in find_destruction(species):
        term = -coeff
        for atom in list(lhs.args):
            term *= atom
        dSdt += term
    
    return dSdt


def get_all_equations(elements):
    rv = []
    for species in elements:
        temp = get_rhs(species)
        rv.append(temp)
    return rv


def rhs(t, state,  element_case, de, T):
    #Element Case 1: elements = [H, Hp, de]
    #Element Case 2: elements = [H, Hp, He, Hep, Hepp, de]
    #Element Case 3: elements = [H, Hp, Hm, He, Hep, Hepp, H2, H2p, grain, de]
    
    if element_case == 1:
        H, Hp = state[0], state[1]
        dnHdt = Hp * de * k2_function(T) - H * de * k1_function(T)
        dnHpdt = H * de * k1_function(T) - Hp * de * k2_function(T)
        return np.array([
            dnHdt, dnHpdt
        ])
        
    elif element_case == 2:
        H, Hp, He, Hep, Hepp = state[0], state[1], state[2], state[3], state[4]
        k1, k2, k3, k4, k5, k6, k7, k8 = k1_function(T),k2_function(T),k3_function(T),k4_function(T),k5_function(T), k6_function(T),k7_function(T),k8_function(T)
                                         
        dnHdt = -H*He*k8 - H*de*k1 + Hp*de*k2
        dnHpdt = H*He*k8 + H*de*k1 + 2*H*k7 - Hp*de*k2
        dnHedt = -He*de*k3 + Hep*de*k4
        dnHepdt = He*de*k3 - Hep*de*k4 - Hep*de*k5 + Hepp*de*k6
        dnHeppdt = Hep*de*k5 - Hepp*de*k6
        return np.array([
            dnHdt, dnHpdt, dnHedt, dnHepdt, dnHeppdt
        ])



def evolve(p0 = 2.04001229e-26,f_H = 1e-6, f_He = 0.0, final_t = 1e10, integrator_type = "lsoda", 
           safety_factor = 1000, element_case = 1, T0 = 1000, path = "Results/"):
    #p0 in units g cm^-3
    
    #Set the elements we will be considering depending on the function input
    if element_case == 1:
        elements = ['H', 'Hp']
    elif element_case == 2:
        elements = ['H', 'Hp', 'He', 'Hep', 'Hepp']
    
    
    #Define Constants
    gamma = (5.0 / 3.0)
    k = 1.38064852 * 10**-16 # cm^2 g K^-1 sec^-2 
    mH = 1.6737236 * 10**-24 # g
    mHe = 6.6464764e-24 # g
    g_to_amu = 1.66054e-24 #grams per a.m.u.
    
    #Define Mass Fractions
    x_H = 0.76
    x_He = 0.24
    
    if element_case == 1:
        #Generate inital number densities
        nHAll = (x_H * p0) / mH
        
        nH = (1-f_H) * nHAll
        nHp = (f_H) * nHAll
        
        de = nHp
        
    elif element_case == 2:
        #Generate inital number densities
        nHAll = (x_H * p0) / mH
        nHeAll = (x_He * p0) / mHe
    
        nH = (1-f_H) * nHAll
        nHp = (f_H) * nHAll
    
        nHe = (1.0-f_He - 0.00001) * nHeAll
        nHep = (f_He) * nHeAll
        nHepp = nHeAll * 0.00001
    
        de = nHp + nHep + 2*nHepp
    
    #Set the initial state depending on which case we are considering
    if element_case == 1:
        initial_state = np.array([nH, nHp])
    elif element_case == 2:
        initial_state = np.array([nH, nHp, nHe, nHep, nHepp])
    
    #Generate an initial mean molecular weight
    u0 = (p0 / (np.sum(initial_state) + de)) / g_to_amu# Divide by g_to_amu to make unitless
    
    #Use u0 and an initial guess for T to generate an initial e
    #This will stay constant throughout the evolution
    #I think the professor told me this should be in erg/g
    e = k * T0 / ((gamma - 1) * u0 * mH) #This e is in cm^2 / sec^2
    
    #Return a value for T given a u using the fact that e will be constant throughout the evolution
    def get_T(u):
        return ((1.0 / k) * e * (gamma - 1) * u * mH)
    
    #Return a value for u given a state vector
    def get_u(state, de):
        u = (p0 / (np.sum(state) + de)) / g_to_amu # Divide by g_to_amu to make unitless
        return (u)
    
    
    #Update the Temperature given a state vector and de. This will be run every time step
    def update(state, de):
        u = get_u(state, de)
        T = get_T(u)
        return (T)
        

    
    #Set up the integrator
    integrator = sint.ode(rhs)
    integrator.set_initial_value(initial_state, t = 0)
    integrator.set_f_params(element_case, de, T0)
    integrator.set_integrator(integrator_type, method = "bdf")
    
    #Set up the data containers to hold results
    state_vector_values = []
    ts = []
    des = []
    ts.append(integrator.t)
    state_vector_values.append(integrator.y)
    des.append(de)
    
    #Initialize a counter to help with progress checking and debugging
    count = 0
    
    #Set the new T to be equal to the initial T. This will change every time step
    T_new = T0
    
    #Define dt as a constant value since I couldn't get the change every timestep method working
    dt = final_t / safety_factor
    
    while integrator.t < final_t:
        #Calculate a new dt (1e-4 is used as a softening length to make sure dt doesn't blow up)
        #Couldn't get this to work
        #dt = safety_factor * abs(np.min(integrator.y / (1e-4 + rhs(integrator.t, integrator.y,
                                                                            #element_case, des[-1], T_new))))
        #Progress Check
        #if count % 1000 == 0:
            #print(integrator.t, final_t)

        #Integrate
        integrator.integrate(integrator.t + dt)
        ts.append(integrator.t)
        state_vector_values.append(integrator.y)
        
        #Update de:
        if element_case == 1:
            des.append(integrator.y[1])
        elif element_case == 2:
            des.append(integrator.y[1] + integrator.y[3] + 2*integrator.y[4])
        
        #Update the Temperature based on the new state
        T_new = update(integrator.y, des[-1])
        
        integrator.set_f_params(element_case, des[-1], T_new)
        
        count += 1
    
    state_vector_values = np.array(state_vector_values)
    ts = np.array(ts)
    
    #Showing Number Densities:
    plt.figure(1)
    
    for i in range(len(elements)):
        plt.loglog(ts,state_vector_values[:,i], label = str(elements[i]) )
    
    plt.loglog(ts, des, label = "de", ls = "-.")
    
    plt.legend()
    
    plt.title("elecase_" + str(element_case) + "_f_H_" + str(f_H) + "_T0_" + str(T0))
    plt.xlabel("Time")
    plt.ylabel("Number Density[Particle / cm^3]")

    plt.savefig(path + "elecase_" + str(element_case) + "_f_H_" + str(f_H) + "_T0_" + str(T0) + "_numden.png")
    plt.clf()
    
    #Showing Fractions:
    if element_case == 1:
        nHall = state_vector_values[:,0] + state_vector_values[:,1]
            
        fH = state_vector_values[:,0] / nHAll
        fHp = state_vector_values[:,1] / nHAll
        
        fs = np.zeros((len(fH), len(elements)))
        
        fs[:,0] = fH
        fs[:,1] = fHp
        
    elif element_case == 2:
            
        nHall = state_vector_values[:,0] + state_vector_values[:,1]
        nHeAll = state_vector_values[:,2] + state_vector_values[:,3] + state_vector_values[:,4]
    
        fH = state_vector_values[:,0] / nHAll
        fHp = state_vector_values[:,1] / nHAll
    
        f_He = state_vector_values[:,2] / nHeAll
        f_Hep = state_vector_values[:,3] / nHeAll
        f_Hepp = state_vector_values[:,4] / nHeAll
                
        fs = np.zeros((len(fH), len(elements)))
    
        fs[:,0] = fH
        fs[:,1] = fHp
        fs[:,2] = f_He
        fs[:,3] = f_Hep
        fs[:,4] = f_Hepp
    
    #plt.figure(2)
    
    #Plot all of the elements except de
    for i in range(len(elements)):
        if elements[i] == "He" or elements[i] == "Hep" or elements[i] == "Hepp":
            linestyle = "--"
        else:
            linestyle = "-"
        plt.loglog(ts,fs[:,i], label = str(elements[i]), ls = linestyle)
    plt.legend()
    
    plt.title("elecase_" + str(element_case) + "_f_H_" + str(f_H) + "_T0_" + str(T0))
    plt.xlabel("Time")
    plt.ylabel("Fraction of Species")

    plt.savefig(path + "elecase_" + str(element_case) + "_f_H_" + str(f_H) + "_T0_" + str(T0) + "_frac.png")
    plt.clf()
    
    #plt.show()




#evolve(element_case=2, f_H = 0.5, f_He = 0.0, safety_factor=1000000, T0 = 10**4)


#def evolve(p0 = 2.04001229e-26,f_H = 1e-6, f_He = 0.0, final_t = 1e10, integrator_type = "lsoda", 
           #safety_factor = 1000, element_case = 1, T0 = 1000):

element_cases = [1,2]
fHs = [0.0, 1e-6, 1.0]
Ts = [10**i for i in range(2,7)]

for i in range(len(element_cases)):
	for j in range(len(fHs)):
		for k in range(len(Ts)):
			evolve(element_case = element_cases[i], f_H = fHs[j], T0 = Ts[k], safety_factor=1000000)
			print("Finished: " + str(element_cases[i]) + " " + str(fHs[j]) + " " + str(Ts[k]))





