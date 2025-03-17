import numpy as np
import matplotlib.pyplot as plt

# Historical EU ETS Prices (in €/ton CO₂)
years = np.array([2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 
                  2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025])
prices = np.array([22, 18, 0.7, 22, 13.3, 14.5, 13.8, 7.6, 4.5, 6.0, 
                   7.5, 5.3, 5.8, 15.9, 24.8, 25, 53.5, 80, 85, 90, 95])

# Future years extended to 2060
future_years_extended = np.arange(2026, 2061)
last_price = prices[-1]  # Price in 2025

# Scenarios
low_scenario_extended = np.full_like(future_years_extended, last_price)  # Constant price
mid_scenario_extended = last_price + 5 * (future_years_extended - 2025)  # +5 €/year
high_scenario_extended = last_price + 10 * (future_years_extended - 2025)  # +10 €/year

# Plot
plt.figure(figsize=(10, 5))
plt.plot(years, prices, 'ko-', label="Historical ETS prices (fossil)", markersize=5)  # Black line for historical data
plt.plot(future_years_extended, low_scenario_extended, 'k--', label="Low (+0 €/yr)")  # Red dashed line
plt.plot(future_years_extended, mid_scenario_extended, 'k--', label="Middle (+5 €/yr)")  # Red dashed line
plt.plot(future_years_extended, high_scenario_extended, 'k--', label="High (+10 €/yr)")  # Red dashed line

# Add constant green lines at 50 €/t and 300 €/t
plt.axhline(y=50, color='g', linestyle='-', label="CRC lower bound (€50/t)")
plt.axhline(y=300, color='g', linestyle='-', label="CRC upper bound (€300/t)")

# Add a purple line for "Auction subsidy Stockholm Exergi" starting in 2028
plt.plot([2028, 2043], [160, 160], 'm-', linewidth=2, label="Auction subsidy Stockholm Exergi (€160/t)")

# Labels and legend
plt.xlabel("Year")
plt.ylabel("Carbon prices (€/ton CO₂)")
plt.title("Alternative Carbon Prices and Future Scenarios (Extended to 2060)")
plt.legend()
plt.grid(True)
plt.show()
