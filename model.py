import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyXSteam.XSteam import XSteam
from scipy.interpolate import LinearNDInterpolator
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

class DecidedTech:
    def __init__(self, name, Qfuel=0, Qnet=0, P=0, memitted=0, mcaptured=0, operating=0):
        self.name = name
        self.Qfuel = Qfuel
        self.Qnet = Qnet
        self.P = P
        self.memitted = memitted
        self.mcaptured = mcaptured
        self.operating = operating

    def print(self):
        """Prints the attributes of the object in a formatted table with units."""
        data = [
            ["Qfuel", f"{self.Qfuel:.2f}", "[MW]"],
            ["Qnet", f"{self.Qnet:.2f}", "[MW]"],
            ["P", f"{self.P:.2f}", "[MW]"],
            ["eta", f"{(self.P+self.Qnet)/self.Qfuel:.2f}", "[-]"],
            ["memitted", f"{self.memitted:.2f}", "[kgCO2/s]"],
            ["mcaptured", f"{self.mcaptured:.2f}", "[kgCO2/s]"],
            ["operating", f"{self.operating:.0f}", "[h/yr]"],
        ]

        print(f"\n{'-'*30}")
        print(f"{self.name:^30}")  # Centered name
        print(f"{'-'*30}")
        print(f"{'Parameter':<12}{'Value':>10}  {'Unit'}")
        print(f"{'-'*30}")

        for row in data:
            print(f"{row[0]:<12}{row[1]:>10}  {row[2]}")

        print(f"{'-'*30}\n")

def create_interpolators(aspen_df):
    # extract 'Flow' and 'Rcapture' columns as x values, the rest are y values
    x1 = aspen_df['Flow']
    x2 = aspen_df['Rcapture']
    x_values = np.column_stack((x1, x2))

    y_values = aspen_df.drop(columns=['Flow', 'Rcapture']).values  
    aspen_interpolators = {}

    for idx, column_name in enumerate(aspen_df.drop(columns=['Flow', 'Rcapture']).columns):
        y = y_values[:, idx]
        interp_func = LinearNDInterpolator(x_values, y)
        aspen_interpolators[column_name] = interp_func

    return aspen_interpolators

def estimate_nominal_cycle(Qnet, P, Qfuel, LHV, psteam, Tsteam, isentropic):
    mfuel = Qfuel/LHV
    HHV = LHV*1.15
    Qfgc = mfuel*HHV - Qfuel

    live = State("live", psteam, Tsteam)
    max_iterations = 1000
    pcond_guess = psteam
    Qestimated = 0
    i = 0
    tol = 0.03
    while abs(Qestimated - Qnet) > Qnet*tol and i < max_iterations:
        pcond_guess = pcond_guess - 0.3
        mix_is = State("mix_is", p=pcond_guess, s=live.s, mix=True)
        hmix = live.h - isentropic*(live.h - mix_is.h)
        boiler = State("boiler", pcond_guess, satL=True)
        msteam = Qfuel/(live.h - boiler.h)
        Qcond = msteam*(hmix - boiler.h)
        Qestimated = Qcond + Qfgc
        Pestimated = msteam*(live.h - hmix)
        i += 1
    if i == max_iterations:
        raise ValueError("Couldn't estimate Rankine cycle!")

    states = {"boiler": boiler, "mix_is": mix_is, "live": live}

    if msteam is not None and Qestimated is not None and pcond_guess > 0:
           return Qfuel, Qcond, Qfgc, Qnet, Pestimated, states
    else:
        raise ValueError("One or more of the variables (msteam, Pestimated, Qfuel, pcond_guess) is not positive.")

def BECCS_stress( 
    Qnet = 140,          #[MW] net district heating, NOTE roughly 40MW should be FGC?
    P = 48.3,            #[MW] net power, or 54.7 gross power ~ my rankine cycle!
    Qfuel = 174,         #[MW] LHV
    LHV = 10.44,         #[MJ/kg] return wood, shredded wood, GROT @39.1%moisture
    psteam = 95,         #[bar]
    Tsteam = 525,        #[C]
    isentropic=0.85,

    i=0.075,
    t=25, # technical is >20, economic is =20
    celc=40,
    cheat=0.80,
    cbio=99999999,

    technology = ["amine","oxy","clc"],
    rate = 0.90, # "high rates needed" (Ramboll Design), so maybe 86-94%?
    operating_increase = [0, 600, 1200],
    timing = [5,10,15,20,25],

    interpolators = None
):
    operating = 4500    #[h]
    t0 = 2027           #[year]
    Tin = 42.6          #[C] 35-50C
    Tout = 88.6         #[C] 70-105C

    # Determining reference case energy balance
    Qfuel, Qcond, Qfgc, Qnet, P, states = estimate_nominal_cycle(Qnet, P, Qfuel, LHV, psteam, Tsteam, isentropic)
    print(Qfgc)
    mfuel = Qfuel/LHV           #[kgf/s]
    memitted = 1.1024 * mfuel   #[kgCO2/s]
    mcaptured = 0
    Pparasitic = 6.4
    REF = DecidedTech("ref", Qfuel, Qnet, P-Pparasitic, memitted, mcaptured, operating)
    REF.print()

    # Determining amine case (with HR) energy balance
    mfluegas = 5.044 * mfuel    #[kg/s]
    Vfluegas = 3.982 * mfuel    #[Nm3/s]

    mcaptured = memitted * rate #[kgCO2/s]
    memitted = memitted * (1-rate)
    Qreb = 2.24 * mcaptured     #[MW]
    print(Qreb)

    Pgross = 50.1/37.1 * Qreb
    Pparasitic = 18.1/37.1 * Qreb 
    Pnet = Pgross - Pparasitic

    Qcond = 73.7/37.1 * Qreb
    Qrec = (11+21.7)/37.1 * Qreb
    Qnet = Qcond + Qrec + Qfgc
    operating += operating_increase[0]
    AMINE = DecidedTech("amine", Qfuel, Qnet, Pnet, memitted, mcaptured, operating)
    AMINE.print()

    regret = 1
    return regret


if __name__ == "__main__":

    aspen_df = pd.read_csv("amine.csv", sep=";", decimal=',')
    aspen_interpolators = create_interpolators(aspen_df)

    regret = BECCS_stress(interpolators = aspen_interpolators)
    print(regret)