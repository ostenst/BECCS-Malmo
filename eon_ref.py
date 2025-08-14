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
    
    # Extract the cbio, celc, Time, and cheat columns from experiments and npv_ref, npv_amine from outcomes
    cbio_values = experiments['cbio'].values
    celc_values = experiments['celc'].values
    time_values = experiments['Time'].values
    cheat_values = experiments['cheat'].values
    npv_ref_values = outcomes['npv_ref'].values
    npv_amine_values = outcomes['npv_amine'].values
    
    # Use only 10% of the data for faster plotting
    sample_size = int(len(cbio_values) * 0.80)
    indices = np.random.choice(len(cbio_values), sample_size, replace=False)
    
    cbio_values = cbio_values[indices]
    celc_values = celc_values[indices]
    time_values = time_values[indices]
    cheat_values = cheat_values[indices]
    npv_ref_values = npv_ref_values[indices]
    npv_amine_values = npv_amine_values[indices]
    
    print(f"Using {sample_size} data points (10% of total)")
    
    # Calculate the energy price ratio (celc/cbio)
    price_ratio = celc_values / cbio_values
    
    print(f"Number of data points: {len(cbio_values)}")
    print(f"cbio range: {cbio_values.min():.2f} to {cbio_values.max():.2f}")
    print(f"celc range: {celc_values.min():.2f} to {celc_values.max():.2f}")
    print(f"Time values: {np.unique(time_values)}")
    print(f"Price ratio (celc/cbio) range: {price_ratio.min():.3f} to {price_ratio.max():.3f}")
    print(f"npv_ref range: {npv_ref_values.min():.2f} to {npv_ref_values.max():.2f}")
    
    # Create the first scatter plot: cbio vs npv_ref with color coding based on Time values
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    
    # Get unique time values and create a color map
    unique_times = np.unique(time_values)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_times)))
    
    # Create scatter plot with different colors for each time period
    for i, time_val in enumerate(unique_times):
        mask = time_values == time_val
        plt.scatter(cbio_values[mask], npv_ref_values[mask], 
                   alpha=0.6, s=50, edgecolors='black', linewidth=0.5, 
                   color=colors[i], label=f'Time: {time_val}')
    
    # Add trend line
    z = np.polyfit(cbio_values, npv_ref_values, 1)
    p = np.poly1d(z)
    plt.plot(cbio_values, p(cbio_values), "r--", alpha=0.8, linewidth=2, label=f'Trend line')
    
    # Customize the plot
    plt.xlabel('cbio (SEK/tonne)', fontsize=12)
    plt.ylabel('npv_ref (million SEK)', fontsize=12)
    plt.title('Impact of cbio (biomass price) on npv_ref\n(Colored by Time)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add correlation coefficient
    correlation_cbio = np.corrcoef(cbio_values, npv_ref_values)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {correlation_cbio:.3f}', 
             transform=plt.gca().transAxes, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Create the second scatter plot: price ratio vs npv_ref with color coding based on Time values
    plt.subplot(1, 3, 2)
    
    # Create scatter plot with different colors for each time period (same color scheme as first plot)
    for i, time_val in enumerate(unique_times):
        mask = time_values == time_val
        plt.scatter(price_ratio[mask], npv_ref_values[mask], 
                   alpha=0.6, s=50, edgecolors='black', linewidth=0.5, 
                   color=colors[i], label=f'Time: {time_val}')
    
    # Add trend line for price ratio
    z_ratio = np.polyfit(price_ratio, npv_ref_values, 1)
    p_ratio = np.poly1d(z_ratio)
    plt.plot(price_ratio, p_ratio(price_ratio), "r--", alpha=0.8, linewidth=2, label=f'Trend line')
    
    # Customize the plot
    plt.xlabel('Energy Price Ratio (celc/cbio)', fontsize=12)
    plt.ylabel('npv_ref (million SEK)', fontsize=12)
    plt.title('Impact of Energy Price Ratio on npv_ref\n(Colored by Time)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add correlation coefficient
    correlation_ratio = np.corrcoef(price_ratio, npv_ref_values)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {correlation_ratio:.3f}', 
             transform=plt.gca().transAxes, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Create the third scatter plot: celc/bio vs celc/cheat with color representing npv_ref
    plt.subplot(1, 3, 3)
    
    # Calculate celc/cheat ratio
    celc_cheat_ratio = celc_values / (celc_values*cheat_values)
    
    # Create scatter plot with color representing npv_ref
    scatter = plt.scatter(price_ratio, celc_cheat_ratio, 
                         c=npv_ref_values, cmap='viridis', 
                         alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('npv_ref (million SEK)', fontsize=12)
    
    # Customize the plot
    plt.xlabel('celc/bio (Energy Price Ratio)', fontsize=12)
    plt.ylabel('celc/cheat', fontsize=12)
    plt.title('celc/bio vs celc/cheat\n(Colored by npv_ref)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation_third = np.corrcoef(price_ratio, celc_cheat_ratio)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {correlation_third:.3f}', 
             transform=plt.gca().transAxes, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plots
    plt.savefig('cbio_and_ratio_vs_npv_ref.png', dpi=300, bbox_inches='tight')
    print("Plots saved as 'cbio_and_ratio_vs_npv_ref.png'")
    
    # Show the plots
    # plt.show()
    
    # Create a new figure with box plots for both npv_ref and npv_amine distributions by celc/cbio ratio ranges
    plt.figure(figsize=(15, 6))
    
    # Define the ratio ranges
    ratio_ranges = [
        (price_ratio < 1, 'celc/cbio < 1'),
        ((price_ratio >= 1) & (price_ratio < 2), '1 ≤ celc/cbio < 2'),
        ((price_ratio >= 2) & (price_ratio <= 5), '2 ≤ celc/cbio ≤ 5')
    ]
    
    # Create subplots for npv_ref and npv_amine
    plt.subplot(1, 2, 1)
    
    # Create box plots for npv_ref
    box_data_ref = []
    labels = []
    
    for mask, label in ratio_ranges:
        if np.any(mask):
            box_data_ref.append(npv_ref_values[mask])
            labels.append(label)
            print(f"{label} (npv_ref): {np.sum(mask)} data points, mean NPV: {npv_ref_values[mask].mean():.2f} million SEK")
    
    # Create the box plot for npv_ref
    bp_ref = plt.boxplot(box_data_ref, tick_labels=labels, patch_artist=True)
    
    # Color the boxes
    colors = ['red', 'yellow', 'green']
    for patch, color in zip(bp_ref['boxes'], colors):
        patch.set_facecolor(color)
    
    # Customize the plot
    plt.ylabel('npv_ref (million SEK)', fontsize=12)
    plt.title('Distribution of npv_ref by Energy Price Ratio (celc/cbio) Ranges', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    plt.text(0.02, 0.98, f'Total data points: {len(npv_ref_values)}', 
             transform=plt.gca().transAxes, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Create subplot for npv_amine
    plt.subplot(1, 2, 2)
    
    # Create box plots for npv_amine
    box_data_amine = []
    
    for mask, label in ratio_ranges:
        if np.any(mask):
            box_data_amine.append(npv_amine_values[mask])
            print(f"{label} (npv_amine): {np.sum(mask)} data points, mean NPV: {npv_amine_values[mask].mean():.2f} million SEK")
    
    # Create the box plot for npv_amine
    bp_amine = plt.boxplot(box_data_amine, tick_labels=labels, patch_artist=True)
    
    # Color the boxes (same colors for consistency)
    for patch, color in zip(bp_amine['boxes'], colors):
        patch.set_facecolor(color)
    
    # Customize the plot
    plt.ylabel('npv_amine (million SEK)', fontsize=12)
    plt.title('Distribution of npv_amine by Energy Price Ratio (celc/cbio) Ranges', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    plt.text(0.02, 0.98, f'Total data points: {len(npv_amine_values)}', 
             transform=plt.gca().transAxes, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Save the box plot figure
    plt.savefig('npv_distribution_by_ratio_ranges.png', dpi=300, bbox_inches='tight')
    print("Box plots saved as 'npv_distribution_by_ratio_ranges.png'")
    
    # Show the box plots
    plt.show()
    
    # Print some statistics
    print("\n=== STATISTICS ===")
    print(f"Mean cbio: {cbio_values.mean():.2f} SEK/tonne")
    print(f"Mean celc: {celc_values.mean():.2f} SEK/MWh")
    print(f"Mean price ratio: {price_ratio.mean():.3f}")
    print(f"Mean npv_ref: {npv_ref_values.mean():.2f} million SEK")
    print(f"Correlation cbio vs npv_ref: {correlation_cbio:.3f}")
    print(f"Correlation price ratio vs npv_ref: {correlation_ratio:.3f}")
    
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
    
    print(f"High price ratio (>75th percentile) NPVs: {npv_ref_values[high_ratio].mean():.2f} million SEK")
    print(f"Low price ratio (<25th percentile) NPVs: {npv_ref_values[low_ratio].mean():.2f} million SEK")

if __name__ == "__main__":
    main()
