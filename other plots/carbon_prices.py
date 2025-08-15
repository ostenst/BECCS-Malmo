import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Use default matplotlib style
plt.style.use("default")

# Historical EU ETS Prices (€/ton CO₂)
years = np.array([2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 
                  2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025])
prices = np.array([22, 18, 0.7, 22, 13.3, 14.5, 13.8, 7.6, 4.5, 6.0, 
                   7.5, 5.3, 5.8, 15.9, 24.8, 25, 53.5, 80, 85, 90, 95])

# Future years and scenario data
future_years_extended = np.arange(2026, 2056)
last_price = prices[-1]

low_scenario = np.full_like(future_years_extended, last_price)
mid_scenario = last_price + 5 * (future_years_extended - 2025)
high_scenario = last_price + 10 * (future_years_extended - 2025)

fig, ax = plt.subplots(figsize=(8.5, 5))

# Scenario shading
ax.fill_between(future_years_extended, low_scenario, high_scenario, 
                color='gray', alpha=0.4)

# Green CRC range lines from 2025 onward with triangle markers
x_vals = np.arange(2025, 2056, 3)  # markers every 3 years

y_vals_low = np.full_like(x_vals, 25)
ax.plot(x_vals, y_vals_low, color='mediumseagreen', lw=1.5, marker='^', markersize=6, label='_nolegend_')

y_vals_high = np.full_like(x_vals, 300)
ax.plot(x_vals, y_vals_high, color='mediumseagreen', lw=1.5, marker='v', markersize=6, label='_nolegend_')

# Gray ETS cap band from 2025 onward
ax.fill_between([2030, 2056], 200, 350, color='gray', alpha=0.2, label="ETS cap (€200–350)")
ax.plot([2030, 2056], [200, 200], color='gray', linestyle='-', linewidth=1.5)
ax.plot([2030, 2056], [350, 350], color='gray', linestyle='-', linewidth=1.5)

# Historical prices
ax.plot(years, prices, 'ko-', label="Historical ETS prices", markersize=5)

# Future scenarios
ax.plot(future_years_extended, low_scenario, 'k--', label="Low (+0 €/yr)")
ax.plot(future_years_extended, mid_scenario, 'k--', label="Middle (+5 €/yr)")
ax.plot(future_years_extended, high_scenario, 'k--', label="High (+10 €/yr)")

# Subsidy line
ax.plot([2028, 2043], [160, 160], color='crimson', linestyle='-', linewidth=2.0,
        label="Auction subsidy Stockholm Exergi (€160/t)")

# Labels and title with adjusted font sizes
ax.set_xlabel("Year", fontsize=13)
ax.set_ylabel("Carbon prices (€/ton CO₂)", fontsize=13)
ax.set_title("Alternative Carbon Prices and Future Scenarios (Extended to 2060)", fontsize=14)

# Adjust tick label font sizes
ax.tick_params(axis='both', labelsize=11)

# Custom legend entry for CRC range
custom_lines = [
    Line2D([0], [0], color='mediumseagreen', lw=1.5, label='CRC range (€25–300)')
]
existing_handles, existing_labels = ax.get_legend_handles_labels()
ax.legend(handles=existing_handles + custom_lines, loc='upper left', fontsize=11)

# Grid
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

# X-axis limit
ax.set_xlim(2005, 2052)

plt.tight_layout()
plt.savefig("carbon_prices", dpi=600)
plt.show()
