import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Read the CSV files
    print("Reading experiments.csv...")
    experiments = pd.read_csv('experiments.csv')
    
    print("Reading outcomes.csv...")
    outcomes = pd.read_csv('outcomes.csv')
    
    print(f"Experiments shape: {experiments.shape}")
    print(f"Outcomes shape: {outcomes.shape}")
    
    # Check if the number of rows match
    if len(experiments) != len(outcomes):
        print(f"Warning: Number of rows don't match! Experiments: {len(experiments)}, Outcomes: {len(outcomes)}")
        # Use the minimum length to ensure alignment
        min_length = min(len(experiments), len(outcomes))
        experiments = experiments.iloc[:min_length]
        outcomes = outcomes.iloc[:min_length]
        print(f"Using first {min_length} rows from both files")
    
    # Extract the cbio, celc, Time, and cheat columns from experiments and npv_ref, npv_amine, regret_1 from outcomes
    cbio_values = experiments['cbio'].values
    celc_values = experiments['celc'].values
    time_values = experiments['Time'].values
    cheat_values = experiments['cheat'].values
    npv_ref_values = outcomes['npv_ref'].values
    npv_amine_values = outcomes['npv_amine'].values
    regret_1_values = outcomes['regret_1'].values
    
    # Use only 10% of the data for faster plotting
    sample_size = int(len(cbio_values) * 0.10)
    indices = np.random.choice(len(cbio_values), sample_size, replace=False)
    
    cbio_values = cbio_values[indices]
    celc_values = celc_values[indices]
    time_values = time_values[indices]
    cheat_values = cheat_values[indices]
    npv_ref_values = npv_ref_values[indices]
    npv_amine_values = npv_amine_values[indices]
    regret_1_values = regret_1_values[indices]
    
    print(f"Using {sample_size} data points (10% of total)")
    
    # Calculate the energy price ratio (celc/cbio)
    price_ratio = celc_values / cbio_values
    
    print(f"Number of data points: {len(cbio_values)}")
    print(f"cbio range: {cbio_values.min():.2f} to {cbio_values.max():.2f}")
    print(f"celc range: {celc_values.min():.2f} to {celc_values.max():.2f}")
    print(f"Time values: {np.unique(time_values)}")
    print(f"Price ratio (celc/cbio) range: {price_ratio.min():.3f} to {price_ratio.max():.3f}")
    print(f"npv_ref range: {npv_ref_values.min():.2f} to {npv_ref_values.max():.2f}")
    
    # Create the first scatter plot: cbio vs npv_amine
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    
    # Create scatter plot with gray color
    plt.scatter(price_ratio, npv_ref_values, 
               alpha=0.6, s=50, edgecolors='black', linewidth=0.5, 
               color='gray')
    
    # Customize the plot
    plt.xlabel('Energy Price Ratio (celc/cbio)', fontsize=12)
    # plt.ylabel('npv_ref (million EUR)', fontsize=12)
    plt.title('Impact of Energy Price Ratio on npv_ref', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='y', labelsize=14)
    plt.tick_params(axis='x', labelsize=14)

    # Create the second scatter plot: price ratio vs npv_ref
    plt.subplot(1, 3, 2)
    
    # Create scatter plot with gray color
    plt.scatter(price_ratio, npv_amine_values, 
               alpha=0.6, s=50, edgecolors='black', linewidth=0.5, 
               color='gray')
    
    # Customize the plot
    plt.xlabel('Energy Price Ratio (celc/cbio)', fontsize=12)
    # plt.ylabel('npv_amine (million EUR)', fontsize=12)
    plt.title('Impact of Energy Price Ratio on npv_amine', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='y', labelsize=14)
    plt.tick_params(axis='x', labelsize=14)
    
    # Create the third scatter plot: price ratio vs regret_1
    plt.subplot(1, 3, 3)
    
    # Create scatter plot with gray color
    plt.scatter(price_ratio, regret_1_values, 
               alpha=0.6, s=50, edgecolors='black', linewidth=0.5, 
               color='gray')
    
    # Customize the plot
    plt.xlabel('Energy Price Ratio (celc/cbio)', fontsize=12)
    # plt.ylabel('regret_1 (million EUR)', fontsize=12)
    plt.title('Impact of Energy Price Ratio on regret_1', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='y', labelsize=14)
    plt.tick_params(axis='x', labelsize=14)
    plt.tight_layout()
    
    # Save the plots
    plt.savefig('npv_regret_eon.png', dpi=600, bbox_inches='tight')

    # Create a new figure with box plots for npv_ref, npv_amine, and regret_1 distributions by celc/cbio ratio ranges
    plt.figure(figsize=(22, 6))
    
    # Define the ratio ranges
    ratio_ranges = [
        (price_ratio < 1, 'celc/cbio < 1'),
        ((price_ratio >= 1) & (price_ratio < 2), '1 ≤ celc/cbio < 2'),
        ((price_ratio >= 2) & (price_ratio <= 5), '2 ≤ celc/cbio ≤ 5')
    ]
    
    # Create subplots for npv_ref, npv_amine, and regret_1
    plt.subplot(1, 3, 1)
    
    # Create box plots for npv_ref
    box_data_ref = []
    labels = []
    
    for mask, label in ratio_ranges:
        if np.any(mask):
            box_data_ref.append(npv_ref_values[mask])
            labels.append(label)
            print(f"{label} (npv_ref): {np.sum(mask)} data points, mean NPV: {npv_ref_values[mask].mean():.2f} million EUR")
    
    # Create the box plot for npv_ref
    bp_ref = plt.boxplot(box_data_ref, tick_labels=labels, patch_artist=True)
    
    # Color the boxes
    colors = ['red', 'yellow', 'green']
    for patch, color in zip(bp_ref['boxes'], colors):
        patch.set_facecolor(color)
    
    # Customize the plot
    # plt.ylabel('npv_ref (million EUR)', fontsize=12)
    plt.title('Distribution of npv_ref by Energy Price Ratio (celc/cbio) Ranges', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Create subplot for npv_amine
    plt.subplot(1, 3, 2)
    
    # Create box plots for npv_amine
    box_data_amine = []
    
    for mask, label in ratio_ranges:
        if np.any(mask):
            box_data_amine.append(npv_amine_values[mask])
            print(f"{label} (npv_amine): {np.sum(mask)} data points, mean NPV: {npv_amine_values[mask].mean():.2f} million EUR")
    
    # Create the box plot for npv_amine
    bp_amine = plt.boxplot(box_data_amine, tick_labels=labels, patch_artist=True)
    
    # Color the boxes (same colors for consistency)
    for patch, color in zip(bp_amine['boxes'], colors):
        patch.set_facecolor(color)
    
    # Customize the plot
    # plt.ylabel('npv_amine (million EUR)', fontsize=12)
    plt.title('Distribution of npv_amine by Energy Price Ratio (celc/cbio) Ranges', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Create subplot for regret_1
    plt.subplot(1, 3, 3)
    
    # Create box plots for regret_1
    box_data_regret = []
    densities = []
    
    for mask, label in ratio_ranges:
        if np.any(mask):
            regret_data = regret_1_values[mask]
            box_data_regret.append(regret_data)
            
            # Calculate density: ratio of positive regret values to total data points
            positive_count = np.sum(regret_data > 0)
            total_count = len(regret_data)
            density = positive_count / total_count
            densities.append(density)
            
            print(f"{label} (regret_1): {total_count} data points, mean regret: {regret_data.mean():.2f} million EUR, density: {density:.3f}")
    
    # Create the box plot for regret_1 with reduced spacing
    bp_regret = plt.boxplot(box_data_regret, tick_labels=labels, patch_artist=True, widths=0.5)
    
    # Color the boxes using coolwarm colormap based on density
    cmap = plt.cm.coolwarm
    for i, (patch, density) in enumerate(zip(bp_regret['boxes'], densities)):
        # Normalize density to [0, 1] for colormap
        color = cmap(density)
        patch.set_facecolor(color)
    
    # Color the median lines black
    for median in bp_regret['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    
    # Customize the plot
    # plt.ylabel('regret_1 (million EUR)', fontsize=12)
    plt.title('Distribution of regret_1 by Energy Price Ratio (celc/cbio) Ranges\n(Colored by density of positive regret)', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    
    plt.tight_layout()
    
    # Save the box plot figure
    plt.savefig('npv_distribution_by_ratio_ranges.png', dpi=300, bbox_inches='tight')
    print("Box plots saved as 'npv_distribution_by_ratio_ranges.png'")
    
    # Save the regret box plot separately
    plt.savefig('regret_box.png', dpi=400, bbox_inches='tight')
    print("Regret box plot saved as 'regret_box.png'")
    
    
    # Print some statistics
    print("\n=== STATISTICS ===")
    print(f"Mean cbio: {cbio_values.mean():.2f} EUR/tonne")
    print(f"Mean celc: {celc_values.mean():.2f} EUR/MWh")
    print(f"Mean price ratio: {price_ratio.mean():.3f}")
    print(f"Mean npv_ref: {npv_ref_values.mean():.2f} million EUR")
    
    # Check for any outliers or interesting patterns
    print(f"\n=== DATA INSIGHTS ===")
    print(f"Number of positive npv_ref values: {np.sum(npv_ref_values > 0)}")
    print(f"Number of negative npv_ref values: {np.sum(npv_ref_values < 0)}")
    print(f"Percentage of positive NPVs: {100 * np.sum(npv_ref_values > 0) / len(npv_ref_values):.1f}%")
    
    # Additional insights about the price ratio
    print(f"\n=== PRICE RATIO INSIGHTS ===")
    print(f"Price ratio quartiles:")
    print(f"  25th percentile: {np.percentile(price_ratio, 25):.3f}")
    print(f"  50th percentile (median): {np.percentile(price_ratio, 50):.3f}")
    print(f"  75th percentile: {np.percentile(price_ratio, 75):.3f}")
    
    # Check if there's a threshold effect
    high_ratio = price_ratio > np.percentile(price_ratio, 75)
    low_ratio = price_ratio < np.percentile(price_ratio, 25)
    
    print(f"High price ratio (>75th percentile) NPVs: {npv_ref_values[high_ratio].mean():.2f} million EUR")
    print(f"Low price ratio (<25th percentile) NPVs: {npv_ref_values[low_ratio].mean():.2f} million EUR")

    plt.show()

if __name__ == "__main__":
    main()
