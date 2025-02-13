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

class ConversionTech:
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

def regret_BECCS( 
    Qnet = 140,          #[MW] net district heating
    P = 48.3,            #[MW] net power
    Qfuel = 174,         #[MW] LHV
    LHV = 10.44,         #[MJ/kg] return wood, shredded wood, GROT @39.1%moisture
    psteam = 95,         #[bar]
    Tsteam = 525,        #[C]
    isentropic=0.85,

    i=0.075,
    t=25, # technical is >20, economic is =20
    celc=40,
    cheat=0.80,
    cbio=35,
    CEPCI=800, 
    sek=0.089,
    usd=0.96,

    technology = ["ref", "amine","oxy","clc"],
    rate = 0.90, # "high rates needed" (Ramboll Design), so maybe 86-94%?
    operating_increase = [0, 600, 1200],
    timing = [0,5,10,15,20], # represents when C&L+amines+ASUs are built, and T&S are paid for, and revenues gained!

    interpolators = None
):
    economic_assumptions = {
        "i": i,
        "t": t,  # technical is >20, economic is =20
        "celc": celc,
        "cheat": cheat,
        "cbio": cbio,
        "CEPCI": CEPCI,
        "sek": sek, #SEK=>EUR
        "usd": usd  #USD=>EUR
    }

    operating = 4500    #[h]
    # t0 = 2027           #[year]
    # Tin = 42.6          #[C] 35-50C
    # Tout = 88.6         #[C] 70-105C
    O2eff = 0.90        #[-] for CLC
    Wasu = 230*3.6      #[MJ/tO2], ref. is the macroscopic study

    # Determining reference case energy balance
    print("Maybe remove estimate_nominal_cycle() if it is irrelevant")
    Qfuel, Qcond, Qfgc, Qnet, P, states = estimate_nominal_cycle(Qnet, P, Qfuel, LHV, psteam, Tsteam, isentropic)
    mfuel = Qfuel/LHV           #[kgf/s]
    memitted = 1.1024 * mfuel   #[kgCO2/s]
    mcaptured = 0
    Pparasitic = 6.4
    REF = ConversionTech("ref", Qfuel, Qnet, P-Pparasitic, memitted, mcaptured, operating)
    REF.print()

    # Determining amine case (with HR) energy balance
    mfluegas = 5.044 * mfuel    #[kg/s]
    Vfluegas = 3.982 * mfuel    #[Nm3/s]

    mcaptured = memitted * rate #[kgCO2/s]
    memitted = memitted * (1-rate)
    Qreb = 2.24 * mcaptured     #[MW]

    Pnet = 31.8/37.1 * Qreb 
    Qcond = 73.7/37.1 * Qreb
    Qrec = (11+21.7)/37.1 * Qreb
    Qnet = Qcond + Qrec + Qfgc
    operating += operating_increase[0]
    AMINE = ConversionTech("amine", Qfuel, Qnet, Pnet, memitted, mcaptured, operating)
    AMINE.print()

    # Determining CLC energy balance
    O2demand = 0.024045 * mfuel #[kmolO2/s]
    LHVO2 = LHV/0.024045        #[MJ/kmolO2] 
    O2oc = O2demand * O2eff
    O2oxy = O2demand * (1-O2eff)

    dHox = 479                  #[MJ/kmolO2], released in AR during oxidation of OC
    dHred = LHVO2 - dHox        #[MJ/kmolO2], released in FR during reduction of OC (positive=>exotherm, otherwise endo)
    Qar = dHox * O2oc           #[MW]
    Qfr = dHred * O2oc
    Qoxy = LHVO2 * O2oxy
    print("CLC heat summarizes to: ", sum([Qar, Qfr, Qoxy]) - Qfuel)

    mCO2 = 1.1024 * mfuel               #[kgCO2/s]
    mH2O = 0.7416 * mfuel               #[kgH2O/s]
    mfluegas = mCO2 + mH2O + O2oxy*32   #[kg/s], inside the post-oxidation chamber (incl. O2oxy)
    mash = 0.013*mfuel
 
    P = REF.P
    Pasu = Wasu/1000*O2oxy*32           #[MW] 
    Pnet = P - Pasu
    mcaptured = mCO2 * rate             #[kgCO2/s], assuming some CO2 is just vented...
    memitted = mCO2 * (1-rate)
    operating += operating_increase[0]

    Afr = 1300 * Qfuel/200
    print("Afr should not be scaled like this!-Magnus")
    CLC = ConversionTech("clc", Qfuel, REF.Qnet , Pnet, memitted, mcaptured, operating)
    CLC.mfluegas = mfluegas
    CLC.print()

    # Determining oxyfuel energy balance
    print("Currently not accounting for reduced oxyfuel-boiler size")
    P = REF.P
    Pasu = Wasu/1000*O2demand*32        #[MW] 
    Pnet = P - Pasu
    mcaptured = mCO2 * rate             #[kgCO2/s], assuming some CO2 is just vented...
    memitted = mCO2 * (1-rate)
    operating += operating_increase[0]
    OXY = ConversionTech("oxy", Qfuel, REF.Qnet , Pnet, memitted, mcaptured, operating)
    OXY.print()

    ### -------------- NEW SECTION ON COSTS AND NPV ------------- ###
    # I need the CAPEX of each ConversionTech.
    # I (later) need transient T&S and CO2 price scenarios. And of CEPCIs???
    # CAPEX: I formulate shopping lists, including the parameter and its base-year CEPCI. NOTE: we exclude most "shared" items, e.g. turbines and FGC

    # Calculating CAPEX [MEUR]:
    REF.shopping_list = {
        'AR' : (0.288*REF.Qfuel+5.08)*usd * CEPCI/576.1
    }
    AMINE.shopping_list = {
        'AR' : (0.288*AMINE.Qfuel+5.08)*usd * CEPCI/576.1,
        'amines' : (2000*sek * AMINE.mcaptured/16.6), # assuming a linear relationship between mcaptured and CAPEX... Let's remove the CL capex cost:
        'CL' : 25.5 * AMINE.mcaptured/37.31    
    }
    print("Remember to subtract the CL")
    CLC.shopping_list = {
        'FR' : (4.98*(Afr/1531)**0.6)*usd * CEPCI/585.7, 
        'AR' : (0.288*CLC.Qfuel+5.08)*usd * CEPCI/576.1,
        'POC' : ( 48.67*10**-6*(CLC.mfluegas) * (1 + np.exp(0.018*(850+273.15)-26.4)) * 1/(0.995-0.98) )*usd * CEPCI/585.7,
        'ASU' : ( 0.02*(59)**0.067/((1-0.95)**0.073) * (O2oxy*1000*3600/453.592)**0.852 )*usd * CEPCI/499.6,
        'OCash' : (6.57*(mash/6)**0.65)*usd * CEPCI/603.1,
    }
    OXY.shopping_list = {
        'AR' : (0.288*OXY.Qfuel+5.08)*usd * CEPCI/576.1,
        'ASU' : ( 0.02*(59)**0.067/((1-0.95)**0.073) * (O2demand*1000*3600/453.592)**0.852 )*usd * CEPCI/499.6,
    }
    TECHS = [REF, AMINE, CLC, OXY]

    print("CAPEX seems ok, but amines are high as they account for ALL plant equipment (unlike the other techs)")
    for tech in TECHS:
        print(f"Technology: {tech.name}")  # Print the technology name
        for item, cost in tech.shopping_list.items():  # Loop through each item and its cost in shopping_list
            print(f"  Item: {item}, Cost: {cost}")
        print("\n")  # Add a newline for better readability

    def calculate_NPV(TECH, economic_assumptions, timing):

        NPV = 1
        return NPV

    for TECH in [REF, AMINE, CLC, OXY]:
        NPV = calculate_NPV(TECH, economic_assumptions, timing)

    regret = 1
    return regret


if __name__ == "__main__":

    aspen_df = pd.read_csv("amine.csv", sep=";", decimal=',')
    aspen_interpolators = create_interpolators(aspen_df)

    regret = regret_BECCS(interpolators = aspen_interpolators)
    print(regret)