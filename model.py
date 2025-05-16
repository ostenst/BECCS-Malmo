import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyXSteam.XSteam import XSteam
from scipy.interpolate import LinearNDInterpolator
import searoute as sr

# Here I insert various helper functions:
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

def regret_BECCS( 
    #Uncertainties:
    O2eff = 0.90,        #[-] for CLC
    Wasu = 230*3.6,      #[MJ/tO2], ref. is the macroscopic study

    operating = 4500,
    dr=0.075,
    lifetime=25, # technical is >20, economic is =20
    celc=40,
    cheat=0.80,
    cbio=25,
    CEPCI=800, 
    sek=0.089,
    usd=0.96,
    ctrans=600,
    cstore=300,
    crc=200,
    cmea=29,    #SEK/kgmea (Ramboll)
    coc=500,    #EUR/tOC Magnus/Felicia

    cAM=2154,   #MSEK , Ramboll Increased capex compared to baseline?
    cFR=0.6,    #[-] Macroscopic exponent
    cASU=0.852, #[-] Macroscopic exponent

    EPC=0.175,
    contingencies=0.25,
    ownercost=0.05,
    overrun=0.30,
    immature=0.50,

    EUA=10, #ETS price increases relative to 75 EUR/t in 2025
    ceiling=250, #an ETS cap relative to DACCS prices

    # Scenarios (all default to False):
    Bioshortage = False,  # True if biomass price increases by 15% per year
    Powersurge = False,   # True if electricity price increases by 20% per year
    Auction = False,      # True if additional revenue from CRC is added
    # Denial = False,
    Integration = True,  
    Capping = True,
    Procurement = True, 
    Time = "Baseline",

    #Levers:
    # decision = "amine", # ["ref", "amine","oxy","clc"],
    rate = 0.90, # "high rates needed" (Ramboll Design), so maybe 86-94%?
    timing = 10, # [5, 10, 15, 20] represents when C&L+amines+ASUs are built, and T&S are paid for, and revenues gained!

    # Constants (put SEK and USD here)
    ETS = 80 # assumed price in 2028
):
    if Time == "Baseline":
        operating_increase = 0
    elif Time == "Downtime":
        operating_increase = -1500
    elif Time == "Uptime":
        operating_increase = 1500
    
    ### -------------- CALCULATING ENERGY AND MASS BALANCES ------------- ###
    LHV = 10.44
    Qfuel = 174.5
    Pnet = 48.3
    Qfgc = 33.3
    Qcond = 106.6
    Qnet = Qcond + Qfgc

    mfuel = Qfuel/LHV           #[kgf/s]
    memitted = 1.0105 * mfuel   #[kgCO2/s]
    mcaptured = 0
    REF = ConversionTech("ref", Qfuel, Qnet, Pnet, memitted, mcaptured, operating)

    # Determining amine case (with HR) energy balance
    mfluegas = 4.952 * mfuel    #[kg/s]
    Vfluegas = 3.982 * mfuel    #[Nm3/s]

    mcaptured = memitted * rate #[kgCO2/s]
    memitted = memitted * (1-rate)

    Ploss_ref = 48.3-31.8       #These are valid for exactly 16.6kgCO2/s, scale them! Check heat balances!
    Qloss_ref = 106.6-73.7
    Pnet -= Ploss_ref/16.6 * mcaptured
    Qcond -= Qloss_ref/16.6 * mcaptured
    Qrec = (11+21.7)/16.6 * mcaptured
    Qnet = Qcond + Qfgc +Qrec        #Qfgc is not scaled - it is constant
    AMINE = ConversionTech("amine", Qfuel, Qnet, Pnet, memitted, mcaptured, operating+operating_increase)

    # Determining C&L balances based on AMINE Ramboll case (although this is already accounted for in the amine balance!)
    Wcompr = 3.5/16.6 * mcaptured #[MW/kgCO2/s * kgCO2/s]
    Qcool  = 3.6/16.6 * mcaptured #[MW/kgCO2/s * kgCO2/s]

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

    mCO2 = 1.0105 * mfuel               #[kgCO2/s]
    mH2O = 0.7416 * mfuel               #[kgH2O/s]
    mfluegas = mCO2 + mH2O + O2oxy*32   #[kg/s], inside the post-oxidation chamber (incl. O2oxy)
    mash = 0.01375*mfuel
 
    P = REF.P
    Pasu = Wasu/1000*O2oxy*32           #[MW] 
    Pnet = P - Pasu - Wcompr - Qcool
    mcaptured = mCO2 * rate             #[kgCO2/s], assuming some CO2 is just vented...
    memitted = mCO2 * (1-rate)

    Vfluegas = mfuel*(2.342 + 4.203)   #[Nm3/s] assuming no O2 in this flue gas... slightly inconsistent with mfluegas
    Across = Vfluegas/5.5                   # Assumed 5.5m/s from Judit
    Afr = 1300/20 * Across                  # Scaled linearly from Anders
    CLC = ConversionTech("clc", Qfuel, REF.Qnet , Pnet, memitted, mcaptured, operating+operating_increase)
    CLC.mfluegas = mfluegas

    # Determining oxyfuel energy balance
    P = REF.P
    Pasu = Wasu/1000*O2demand*32        #[MW], Macroscopic? Or from Anders maybe?
    Pnet = P - Pasu - Wcompr - Qcool
    mcaptured = mCO2 * rate             #[kgCO2/s], assuming some CO2 is just vented...
    memitted = mCO2 * (1-rate)
    OXY = ConversionTech("oxy", Qfuel, REF.Qnet , Pnet, memitted, mcaptured, operating+operating_increase)

    # for tech in [REF,AMINE,OXY,CLC]:
    #     tech.print()

    ### -------------- CALCULATING COSTS AND NPV ------------- ###
    # Calculating CAPEX per item [MEUR]:
    REF.shopping_list = {
    }
    AMINE.shopping_list = {
        'amines' : cAM*sek * AMINE.mcaptured/16.6, # assuming a linear relationship between mcaptured and CAPEX... Let's remove the CL capex cost:
    }
    CLC.shopping_list = {
        'FR' : 4.98*(Afr/1531)**cFR*usd * CEPCI/585.7 *1.4, 
        'cyclone' : 0.345*( 3 )*usd * CEPCI/576.1 *1.4, 
        'POC' : ( 48.67*10**-6*(CLC.mfluegas) * (1 + np.exp(0.018*(850+273.15)-26.4)) * 1/(0.995-0.98) )*usd * CEPCI/585.7 *1.3,
        'ASU' : 0.02*(59)**0.067/((1-0.95)**0.073) * (O2oxy*1000*3600/453.592)**cASU *usd * CEPCI/499.6 *1.3,
        'OCash' : (4.6*(mash/6.7)**0.56)*usd * CEPCI/603.1 *1.2,
        'CL' : 25.5 * mcaptured/37.31 * CEPCI/607.5 *1.3,  #Assuming that Deng had cost year = 2019 NOTE: unclear if installation 1.3 should be included or not? NOTE: Kjästad estimates CL cost in SKEPPKOSTNAD excel?
        'interim' : (53000+2400*(4000)**0.6 )*10**-6 *usd * CEPCI/499.6 *1.2, #Function from Judit, 4000m3 from Ramboll, CEPCI from Google
    }
    OXY.shopping_list = {
        'ASU' : 0.02*(59)**0.067/((1-0.95)**0.073) * (O2demand*1000*3600/453.592)**cASU *usd * CEPCI/499.6 *1.3,
        'CL' : 25.5 * mcaptured/37.31 * CEPCI/607.5 *1.3,  
        'interim' : (53000+2400*(4000)**0.6 )*10**-6 *usd * CEPCI/499.6 *1.2,  
    }

    # Escalating CAPEX of REF and AMINES
    REF.CAPEX = 0
    AMINE.CAPEX = AMINE.shopping_list["amines"]*(1 + overrun)

    # Escalating CAPEX of CLC and OXY
    initial_items = ['FR', 'cyclone', 'POC', 'ASU', 'OCash',] # These should have an additional contingency
    delayed_items = ['CL', 'interim']

    BEC =  sum(value for key, value in CLC.shopping_list.items() if key in initial_items)
    EPCC = BEC*(1 + EPC) # To harmonize with Ramboll amine costs
    TPC = EPCC*(1 + contingencies) # Applying Ramboll's logic
    TOC = TPC*(1 + ownercost)
    CLC.CAPEX_initial = TOC*(1 + immature)*(1 + overrun)

    BEC =  sum(value for key, value in CLC.shopping_list.items() if key in delayed_items)
    EPCC = BEC*(1 + EPC)
    TPC = EPCC*(1 + contingencies)
    CLC.CAPEX = TPC*(1 + ownercost)*(1 + overrun)

    BEC =  sum(OXY.shopping_list.values())
    EPCC = BEC*(1 + EPC) # To harmonize with Ramboll amine costs
    TPC = EPCC*(1 + contingencies) # Applying Ramboll's logic
    OXY.CAPEX = TPC*(1 + ownercost)*(1 + overrun)

    # CAPEX = []
    # for items, contingency_i in [[initial_items, contingency_clc],[delayed_items, contingency_process]]:
    #     BEC =  sum(value for key, value in CLC.shopping_list.items() if key in items)
    #     EPCC = BEC*(1 + EPC)
    #     TPC = EPCC + contingency_i*BEC + contingency_project*(EPCC + contingency_i*BEC)
    #     TOC = TPC*(1 + ownercost)
    #     TCR = 1.154*TOC #Check Macroscopic ref
    #     CAPEX.append(TCR)
    # CLC.CAPEX_initial = CAPEX[0]
    # CLC.CAPEX = CAPEX[1]

    # BEC =  sum(OXY.shopping_list.values())
    # EPCC = BEC*(1 + EPC)
    # TPC = EPCC + contingency_process*BEC + contingency_project*(EPCC + contingency_process*BEC)
    # TOC = TPC*(1 + ownercost)
    # TCR = 1.154*TOC #Check Macroscopic ref
    # OXY.CAPEX = TCR
    
    def calculate_NPV(TECH, cbio, celc, ETS, crc):
        analysis_period = timing + lifetime  # Example: invest after 5, lifetime of 25 => 30 years

        invested = False
        NPV = 0
        
        for t in range(1, analysis_period):
    
            if Bioshortage and t < 11:
                cbio *= 1.10
            if Powersurge and t < 4:
                celc *= 1.20
            if Integration:
                # Increase ETS prices
                if not Capping:
                    ETS += EUA
                elif Capping and ETS < ceiling:
                    ETS += EUA

                # Determine the best CRC price
                if ETS > crc:   
                    crc = ETS
                if ceiling > crc and Capping and Procurement:
                    crc = ceiling

            # Adding CAPEX
            costs = 0
            revenues = 0
            if (t == 1 or t == 2) and TECH.CAPEX_initial is not None:
                costs += TECH.CAPEX_initial / 2  # [MEUR] Assuming 50% of CAPEX for 2 construction years

            if t == timing or t == timing + 1:
                invested = True  # Implies the energy balance (Qdh, Pel, Qfuel) has changed, incl. (C&L and ASU) and that T&S is operated
                costs += TECH.CAPEX / 2  # [MEUR]

            # Adding OPEX and revenues
            if t > timing + 1 and invested:

                # Calculate operational costs and revenues
                costs += TECH.Qfuel * TECH.operating * cbio * 10**-6  # Biomass fuel costs
                revenues += (TECH.Qnet * (cheat * celc) + TECH.P * celc) * TECH.operating * 10**-6  # Revenue from CHP

                costs += TECH.mcaptured / 1000 * 3600 * TECH.operating * (ctrans*sek + cstore*sek) * 10**-6  # Capture and storage costs
                if Auction and t < timing+15+2: #Add two years for the capital delay before operations
                    revenues += TECH.mcaptured / 1000 * 3600 * TECH.operating * (crc+160) * 10**-6  # Revenue from CO2 capture credits
                else:
                    revenues += TECH.mcaptured / 1000 * 3600 * TECH.operating * crc * 10**-6  # Revenue from CO2 capture credits

                if TECH.name == "amine":
                    costs += cmea * sek * 1.5 * TECH.mcaptured / 1000 * 3600 * TECH.operating * 10**-6  # Additional costs for amine capture
                if TECH.name == "clc":
                    costs += 1 / 1000 * TECH.Qfuel * TECH.operating * coc * 10**-6  # Additional costs for chemical-looping

            else:
                costs += REF.Qfuel * REF.operating * cbio * 10**-6  # Reference fuel costs
                revenues += (REF.Qnet * (cheat * celc) + REF.P * celc) * REF.operating * 10**-6  # Reference revenue

            NPV += (revenues - costs) / (1 + dr) ** t

        return NPV

    TECHS = [REF, AMINE, CLC, OXY]
    npv_values = {}
    for tech in TECHS:
        # Reset cbio and celc before each call to calculate_NPV
        initial_cbio = cbio
        initial_celc = celc
        initial_ETS = ETS
        initial_crc = crc
        npv_values[tech.name] = calculate_NPV(tech, initial_cbio, initial_celc, initial_ETS, initial_crc)

    # Recalculate regret values according to new RQs
    regret_1 = npv_values["ref"] - npv_values["amine"] 
    regret_2 = npv_values["oxy"] - npv_values["amine"]
    regret_3 = npv_values["clc"] - npv_values["amine"]

    results = {
        "regret_1" : regret_1, 
        "regret_2": regret_2,
        "regret_3": regret_3,

        "npv_ref" : npv_values["ref"],       
        "npv_amine": npv_values["amine"],     
        "npv_oxy": npv_values["oxy"],     
        "npv_clc": npv_values["clc"],     
    }

    return results


if __name__ == "__main__":

    dict = regret_BECCS()

    print("On this branch, we have three types of regret outputs. So the regret function and outputs need to be adapted.")
    print("amine vs. ref regret=", dict["regret_1"])
    print("amine vs. oxy regret=", dict["regret_2"])
    print("amine vs. clc regret=", dict["regret_3"])

    print("\n I am only missing EU ETS integration scenarios! :) And T&S maybe")