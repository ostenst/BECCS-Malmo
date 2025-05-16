# import numpy as np
# import matplotlib.pyplot as plt

# # Historical EU ETS Prices (in €/ton CO₂)
# years = np.array([2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 
#                   2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025])
# prices = np.array([22, 18, 0.7, 22, 13.3, 14.5, 13.8, 7.6, 4.5, 6.0, 
#                    7.5, 5.3, 5.8, 15.9, 24.8, 25, 53.5, 80, 85, 90, 95])

# # Future years extended to 2060
# future_years_extended = np.arange(2026, 2061)
# last_price = prices[-1]  # Price in 2025

# # Scenarios
# low_scenario_extended = np.full_like(future_years_extended, last_price)  # Constant price
# mid_scenario_extended = last_price + 5 * (future_years_extended - 2025)  # +5 €/year
# high_scenario_extended = last_price + 10 * (future_years_extended - 2025)  # +10 €/year

# # Set seaborn style but override the background
# plt.style.use("bmh")

# # Create figure and axis
# fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")  # Ensure figure background is white
# ax.set_facecolor("white")  # Ensure the plot area is also white

# # Plot data
# plt.plot(years, prices, 'ko-', label="Historical ETS prices (fossil)", markersize=5)  
# plt.plot(future_years_extended, low_scenario_extended, 'k--', label="Low (+0 €/yr)")  
# plt.plot(future_years_extended, mid_scenario_extended, 'k--', label="Middle (+5 €/yr)")  
# plt.plot(future_years_extended, high_scenario_extended, 'k--', label="High (+10 €/yr)")  

# # Add constant green lines at 50 €/t and 300 €/t
# plt.axhline(y=50, color='mediumseagreen', linestyle='-', linewidth=1, label="CRC lower bound (€50/t)")
# plt.axhline(y=300, color='mediumseagreen', linestyle='-', linewidth=1, label="CRC upper bound (€300/t)")

# # Add a subsidy line for Stockholm Exergi
# plt.plot([2028, 2043], [160, 160], color='crimson', linestyle='-', linewidth=1, label="Auction subsidy Stockholm Exergi (€160/t)")

# # Labels and title
# plt.xlabel("Year")
# plt.ylabel("Carbon prices (€/ton CO₂)")
# plt.title("Alternative Carbon Prices and Future Scenarios (Extended to 2060)")

# # Ensure grid lines are visible
# plt.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)  

# # Legend
# plt.legend()

# # Show the plot
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Use default matplotlib style
plt.style.use("default")

# Historical EU ETS Prices (€/ton CO₂)
years = np.array([2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 
                  2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025])
prices = np.array([22, 18, 0.7, 22, 13.3, 14.5, 13.8, 7.6, 4.5, 6.0, 
                   7.5, 5.3, 5.8, 15.9, 24.8, 25, 53.5, 80, 85, 90, 95])

# Future years and scenario data
future_years_extended = np.arange(2026, 2052)
last_price = prices[-1]

low_scenario = np.full_like(future_years_extended, last_price)
mid_scenario = last_price + 5 * (future_years_extended - 2025)
high_scenario = last_price + 10 * (future_years_extended - 2025)

fig, ax = plt.subplots(figsize=(10, 5))

# Shade region between low and high scenarios
ax.fill_between(future_years_extended, low_scenario, high_scenario, 
                color='gray', alpha=0.2, label="Scenario range (Low–High)")

# Shade region between constant green lines
plt.axhline(y=25, color='mediumseagreen', linestyle='-', linewidth=1.5)
plt.axhline(y=300, color='mediumseagreen', linestyle='-', linewidth=1.5)
# ax.axhspan(25, 300, color='mediumseagreen', alpha=0.2, label="CRC price range (€25–300)")

# Shade region between ETS cap
plt.axhline(y=200, color='gray', linestyle='-', linewidth=1)
plt.axhline(y=350, color='gray', linestyle='-', linewidth=1)
ax.axhspan(200, 350, color='gray', alpha=0.2, label="ETS cap (€200–350)")

# Plot lines
ax.plot(years, prices, 'ko-', label="Historical ETS prices", markersize=5)
ax.plot(future_years_extended, low_scenario, 'k--', label="Low (+0 €/yr)")
ax.plot(future_years_extended, mid_scenario, 'k--', label="Middle (+5 €/yr)")
ax.plot(future_years_extended, high_scenario, 'k--', label="High (+10 €/yr)")

# Subsidy line
ax.plot([2028, 2043], [160, 160], color='crimson', linestyle='-', linewidth=1.5,
        label="Auction subsidy Stockholm Exergi (€160/t)")

# Labels and title
ax.set_xlabel("Year")
ax.set_ylabel("Carbon prices (€/ton CO₂)")
ax.set_title("Alternative Carbon Prices and Future Scenarios (Extended to 2060)")

# Grid and legend
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
ax.legend()

plt.tight_layout()
plt.show()
