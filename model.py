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
        self.operating_increase = 0
        self.shopping_list = None
        self.CAPEX_initial = None
        self.CAPEX = None

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
    #Uncertainties:
    Qnet = 140,          #[MW] net district heating
    P = 48.3,            #[MW] net power
    Qfuel = 174,         #[MW] LHV
    LHV = 10.44,         #[MJ/kg] return wood, shredded wood, GROT @39.1%moisture
    psteam = 95,         #[bar]
    Tsteam = 525,        #[C]
    isentropic=0.85,
    O2eff = 0.90,        #[-] for CLC
    Wasu = 230*3.6,      #[MJ/tO2], ref. is the macroscopic study

    dr=0.075,
    lifetime=25, # technical is >20, economic is =20
    celc=40,
    cheat=0.80,
    cbio=25,
    CEPCI=800, 
    sek=0.089,
    usd=0.96,
    ctrans=50,
    cstore=30,
    crc=100,
    cmea=29,    #SEK/kgmea (Ramboll)
    coc=500,    #EUR/tOC Magnus/Felicia

    EPC=0.175,
    contingency_process=0.05,
    contingency_clc=0.40,
    contingency_project=0.20,
    ownercost=0.20,

    #Levers:
    decision = "amine", # ["ref", "amine","oxy","clc"],
    rate = 0.90, # "high rates needed" (Ramboll Design), so maybe 86-94%?
    operating = 4500,
    operating_increase = 600, # [0, 600, 1200],
    timing = 10, # [5, 10, 15, 20] represents when C&L+amines+ASUs are built, and T&S are paid for, and revenues gained!

):
    # Determining reference case energy balance
    # print("Maybe remove estimate_nominal_cycle() if it is irrelevant")
    Qfuel, Qcond, Qfgc, Qnet, P, states = estimate_nominal_cycle(Qnet, P, Qfuel, LHV, psteam, Tsteam, isentropic)
    mfuel = Qfuel/LHV           #[kgf/s]
    memitted = 1.1024 * mfuel   #[kgCO2/s]
    mcaptured = 0
    Pparasitic = 6.4
    REF = ConversionTech("ref", Qfuel, Qnet, P-Pparasitic, memitted, mcaptured, operating)
    # REF.print()

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
    AMINE = ConversionTech("amine", Qfuel, Qnet, Pnet, memitted, mcaptured, operating+operating_increase)
    # AMINE.print()

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
    # print("CLC heat summarizes to: ", sum([Qar, Qfr, Qoxy]) - Qfuel)

    mCO2 = 1.1024 * mfuel               #[kgCO2/s]
    mH2O = 0.7416 * mfuel               #[kgH2O/s]
    mfluegas = mCO2 + mH2O + O2oxy*32   #[kg/s], inside the post-oxidation chamber (incl. O2oxy)
    mash = 0.01375*mfuel
 
    P = REF.P
    Pasu = Wasu/1000*O2oxy*32           #[MW] 
    Pnet = P - Pasu
    mcaptured = mCO2 * rate             #[kgCO2/s], assuming some CO2 is just vented...
    memitted = mCO2 * (1-rate)

    # Afr = 1300 * Qfuel/200
    # print("Afr should not be scaled like this!-Magnus")
    Vfluegas = mfuel*(0.514 + 0.923)   #[Nm3/s] assuming no O2 in this flue gas... slightly inconsistent with mfluegas
    Across = Vfluegas/5.5                   # Assumed 5.5m/s from Judit
    print(Across)
    Afr = 1300/20 * Across                  # Scaled linearly from Anders
    print("Afr is very small, since Vfluegas is very small... how to calculate this? Not just H2O and CO2?")
    CLC = ConversionTech("clc", Qfuel, REF.Qnet , Pnet, memitted, mcaptured, operating+operating_increase)
    CLC.mfluegas = mfluegas
    # CLC.print()

    # Determining oxyfuel energy balance
    # print("Currently not accounting for reduced oxyfuel-boiler size")
    P = REF.P
    Pasu = Wasu/1000*O2demand*32        #[MW] 
    Pnet = P - Pasu
    mcaptured = mCO2 * rate             #[kgCO2/s], assuming some CO2 is just vented...
    memitted = mCO2 * (1-rate)
    OXY = ConversionTech("oxy", Qfuel, REF.Qnet , Pnet, memitted, mcaptured, operating+operating_increase)
    # OXY.print()

    # Determining C&L balances
    Wcompr = 13.17/37.31 * mcaptured #[MW/kgCO2/s * kgCO2/s] Deng's massflow and work
    Qcool  = 43.30/37.31 * mcaptured #[MW/kgCO2/s * kgCO2/s]

    ### -------------- NEW SECTION ON COSTS AND NPV ------------- ###
    # I need the CAPEX of each ConversionTech.
    # I (later) need transient T&S and CO2 price scenarios. And of CEPCIs???
    # CAPEX: I formulate shopping lists, including the parameter and its base-year CEPCI. NOTE: we exclude most "shared" items, e.g. turbines and FGC
    # print("Current logic: the AMINE CAPEX is well-defined, but includes C&L. I thus need to add a CAPEX estimate of C&L to CLC+OXY which is not completely the same as for the AMINES.")
    # print("That is sad. But, the CAPEX of C&L is not significant anyway - the OPEX is what matters. And OPEX will be similar (the same) across technologies!")

    # Calculating CAPEX per item [MEUR]:
    REF.shopping_list = {
        # 'AR' : (0.288*REF.Qfuel+5.08)*usd * CEPCI/576.1
    }
    AMINE.shopping_list = {
        'amines' : (2000*sek * AMINE.mcaptured/16.6), # assuming a linear relationship between mcaptured and CAPEX... Let's remove the CL capex cost:
    }
    CLC.shopping_list = {
        'FR' : (4.98*(Afr/1531)**0.6)*usd * CEPCI/585.7 *1.4, 
        'cyclone' : 0.345*( 3 )*usd * CEPCI/576.1 *1.4, 
        'POC' : ( 48.67*10**-6*(CLC.mfluegas) * (1 + np.exp(0.018*(850+273.15)-26.4)) * 1/(0.995-0.98) )*usd * CEPCI/585.7 *1.3,
        'ASU' : ( 0.02*(59)**0.067/((1-0.95)**0.073) * (O2oxy*1000*3600/453.592)**0.852 )*usd * CEPCI/499.6 *1.3,
        'OCash' : (4.6*(mash/6.7)**0.56)*usd * CEPCI/603.1 *1.2,
        'CL' : 25.5 * mcaptured/37.31 * CEPCI/607.5 *1.3,  #Assuming that Deng had cost year = 2019 NOTE: unclear if installation 1.3 should be included or not?
        'interim' : (53000+2400*(4000)**0.6 )*10**-6 *usd * CEPCI/499.6 *1.2, #Function from Judit, 4000m3 from Ramboll, CEPCI from Google
    }
    OXY.shopping_list = {
        'ASU' : ( 0.02*(59)**0.067/((1-0.95)**0.073) * (O2demand*1000*3600/453.592)**0.852 )*usd * CEPCI/499.6 *1.3,
        'CL' : 25.5 * mcaptured/37.31 * CEPCI/607.5 *1.3,  
        'interim' : (53000+2400*(4000)**0.6 )*10**-6 *usd * CEPCI/499.6 *1.2,  
    }
    # print("Neglected costs for CLC and OXY: flue gas cleaning FGC, molecular sieves, pumps/fans, minor HEXs. But double counting OCash collection for CLC")
    # print("Also escalating the C&L and ASU of CLC way more than for oxyfuel, i.e. 40 vs 5 %")
    # print("Assuming that FGC is roughly equally costly for all options, and that our C&L CAPEX estimation is comparable to the estimation in amine CAPEX")

    # Escalating CAPEX
    REF.CAPEX = 0
    AMINE.CAPEX = sum(AMINE.shopping_list.values())

    initial_items = ['FR', 'cyclone', 'POC', 'OCash']
    delayed_items = ['ASU', 'CL', 'interim']
    CAPEX = []
    for items, contingency_i in [[initial_items, contingency_clc],[delayed_items, contingency_process]]:
        BEC =  sum(value for key, value in CLC.shopping_list.items() if key in items)
        EPCC = BEC*(1 + EPC)
        TPC = EPCC + contingency_i*BEC + contingency_project*(EPCC + contingency_i*BEC)
        TOC = TPC*(1 + ownercost)
        TCR = 1.154*TOC #Check Macroscopic ref
        CAPEX.append(TCR)
    CLC.CAPEX_initial = CAPEX[0]
    CLC.CAPEX = CAPEX[1]

    BEC =  sum(OXY.shopping_list.values())
    EPCC = BEC*(1 + EPC)
    TPC = EPCC + contingency_process*BEC + contingency_project*(EPCC + contingency_process*BEC)
    TOC = TPC*(1 + ownercost)
    TCR = 1.154*TOC #Check Macroscopic ref
    OXY.CAPEX = TCR

    TECHS = [REF,AMINE,CLC,OXY]

    # Calculating NPV regret
    def calculate_NPV(TECH):
        analysis_period = timing + lifetime # Example: invest after 5, lifetime of 25 => 30 years

        invested = False
        NPV = 0
        for t in range(1, analysis_period):

            # Adding CAPEX
            costs = 0
            revenues = 0
            if (t==1 or t==2) and TECH.CAPEX_initial is not None:
                costs += TECH.CAPEX_initial/2 #[MEUR] Assuming 50% of CAPEX for 2 construction years

            if t==timing or t==timing+1:
                invested = True # Implies the energy balance (Qdh, Pel, Qfuel) has changed, incl. (C&L and ASU) and that T&S is operated
                costs += TECH.CAPEX/2 #[MEUR]

            # Adding OPEX and revenues
            if t>timing+1 and invested:
                costs += TECH.Qfuel * TECH.operating * cbio *10**-6 #[MEUR/yr]
                costs += TECH.mcaptured/1000*3600 * TECH.operating * (ctrans + cstore) *10**-6 
                if TECH.name=="amine":
                    costs += cmea*sek *1.5 *TECH.mcaptured/1000*3600 * TECH.operating *10**-6 #1.5 from Ramboll
                if TECH.name=="clc":
                    costs += 1/1000 *TECH.Qfuel* TECH.operating* coc *10**-6 #1kgOC/MWhbr from Magnus
                
                revenues += ( TECH.Qnet*(cheat*celc) + TECH.P*celc )*TECH.operating *10**-6 #[MEUR/yr]
                revenues += TECH.mcaptured/1000*3600 * TECH.operating * crc *10**-6
            else:
                costs += REF.Qfuel * REF.operating * cbio *10**-6 
                revenues += ( REF.Qnet*(cheat*celc) + REF.P*celc )*REF.operating *10**-6 #[MEUR/yr]
            
            NPV += (revenues-costs) / (1+dr)**t

        return NPV

    def calculate_regret(chosen_tech, npv_values):
        max_npv = max(npv_values.values())
        regret = max_npv - npv_values[chosen_tech] 
        return regret

    TECHS = [REF, AMINE, CLC, OXY]
    npv_values = {tech.name: calculate_NPV(tech) for tech in TECHS}
    regret_values = {tech.name: calculate_regret(tech.name, npv_values) for tech in TECHS}

    results = {
        "regret_decision" : regret_values[decision],
        "amine_capex" : AMINE.CAPEX,
        "clc_capex" : CLC.CAPEX,
        "oxy_capex" : OXY.CAPEX,
        "ref_capex" : 0,
        }

    return results


if __name__ == "__main__":

    # aspen_df = pd.read_csv("amine.csv", sep=";", decimal=',')
    # aspen_interpolators = create_interpolators(aspen_df)

    dict = regret_BECCS()
    print("I regret my decision this much in terms of NPV [MEUR]:\n", dict["regret_decision"])

    print("I need to sync this with Fredrik before continuing! To follow the RDM framing phase")
    print("To do that, I need to draw my different cases in detail - showcasing what is _the same_ between each cases")