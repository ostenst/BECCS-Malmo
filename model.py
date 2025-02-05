import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyXSteam.XSteam import XSteam
import searoute as sr

# Here I insert various helper functions:
steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)
class State:
    def __init__(self, Name, p=None, T=None, s=None, satL=False, satV=False, mix=False):
        self.Name = Name
        if satL==False and satV==False and mix==False:
            self.p = p
            self.T = T
            self.s = steamTable.s_pt(p,T)
            self.h = steamTable.h_pt(p,T)
        if satL==True:
            self.p = p
            self.T = steamTable.tsat_p(p)
            self.s = steamTable.sL_p(p)
            self.h = steamTable.hL_p(p) 
        if satV==True:
            self.p = p
            self.T = steamTable.tsat_p(p)
            self.s = steamTable.sV_p(p)
            self.h = steamTable.hV_p(p)
        if mix==True:
            self.p = p
            self.T = steamTable.tsat_p(p)
            self.s = s
            self.h = steamTable.h_ps(p,s)
        if self.p is None or self.T is None or self.s is None or self.h is None:
            raise ValueError("Steam properties cannot be determined")
        
def estimate_nominal_cycle(psteam, Tsteam, Qdh, P):
    live = State("live", psteam, Tsteam)

    Ptarget = P
    max_iterations = 1000
    pcond_guess = psteam
    Pestimated = 0
    i = 0
    tol = 0.05
    while abs(Pestimated - Ptarget) > Ptarget*tol and i < max_iterations:
        pcond_guess = pcond_guess - 0.1
        mix = State("mix", p=pcond_guess, s=live.s, mix=True)
        boiler = State("boiler", pcond_guess, satL=True)
        msteam = Qdh/(mix.h-boiler.h)
        Pestimated = msteam*(live.h-mix.h) # Maybe add an isentropic efficiency here to make the Qfuel(LHV)=>Qdh+Qfgc calculation legitimate? NOTE
        i += 1
    if i == max_iterations:
        raise ValueError("Couldn't estimate Rankine cycle!")

    Qfuel = msteam*(live.h-boiler.h)
    P = Pestimated
    msteam = msteam
    states = {"boiler": boiler, "mix": mix, "live": live}

    if msteam is not None and Pestimated is not None and Qfuel > 0 and pcond_guess > 0:
           return Qfuel, P, msteam, states
    else:
        raise ValueError("One or more of the variables (msteam, Pestimated, Qfuel, pcond_guess) is not positive.")
    
def BECCS_stress( 
    dTreb=10,

    i=0.075,
    t=25, # technical is >20, economic is =20

    celc=40,
    cheat=0.80,
    cbio=99999999,

    technology = ["amine","oxy","clc"],
    rate = 0.90, # "high rates needed" (Ramboll Design), so maybe 86-94%?
    operating_increase = [0, 600, 1200],
    timing = [5,10,15,20,25]

):

    Qdh = 140           #[MW] net district heating, NOTE roughly 40MW should be FGC?
    P = 48.3            #[MW] net power
    Qfuel = 174         #[MW] 
    LHV = 10.44         #[MJ/kg] return wood, shredded wood, GROT @39.1%moisture
    operating = 4500    #[h]
    t0 = 2027           #[year]

    psteam = 95         #[bar]
    Tsteam = 525        #[C]
    Tin = 42.6          #[C] 35-50C
    Tout = 88.6         #[C] 70-105C

    mCO2 = 70           #[t/h] use this with Beiron CAPEX function for amines at CHPs

    Qfuel, P, msteam, states = estimate_nominal_cycle(psteam, Tsteam, Qdh, P)
    print(Qfuel, P, msteam, states)

    regret = 1
    return regret


if __name__ == "__main__":

    regret = BECCS_stress()
    print(regret)