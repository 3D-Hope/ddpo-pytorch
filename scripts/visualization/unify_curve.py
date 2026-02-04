import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--curve_name", type=str, default="normal_compressibility_30_steps")
    parser.add_argument("--csv_path", type=str, default="wandb_export_2026-01-12T14_24_54.954+05_45.csv")
    args = parser.parse_args()
    curve_name = args.curve_name
    csv_path = args.csv_path

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    csv_path = os.path.join(project_root, "curves", args.csv_path)
    df = pd.read_csv(csv_path)

    # Automatically detect all reward_mean columns
    reward_mean_columns = [col for col in df.columns if col.endswith(" - reward_mean")]

    # Sort columns alphabetically
    reward_mean_columns.sort()

    if len(reward_mean_columns) == 0:
        raise ValueError("No reward_mean columns found in the CSV file")

    print(f"Found {len(reward_mean_columns)} component(s): {reward_mean_columns}")

    # Process each component dynamically
    x_unified_list = []
    y_unified_list = []
    x_separators = []  # Store x positions for vertical separator lines
    current_offset = 0

    for col in reward_mean_columns:
        # Extract data for this component
        comp_data = df[["Step", col]].copy()
        comp_data = comp_data[comp_data[col].notna() & (comp_data[col] != "")]
        
        if len(comp_data) == 0:
            print(f"Warning: No data found for {col}, skipping...")
            continue
        
        # Convert to float
        comp_data[col] = comp_data[col].astype(float)
        
        # Get step values
        comp_steps = comp_data["Step"].values
        
        # Calculate step range for this component
        step_range = comp_steps.max() - comp_steps.min() + 1
        
        # Create x-axis values with offset
        x_comp = comp_steps - comp_steps.min() + current_offset
        
        # Extract y values
        y_comp = comp_data[col].values
        
        # Store for concatenation
        x_unified_list.append(x_comp)
        y_unified_list.append(y_comp)
        
        # Store separator position (end of this component)
        x_separators.append(x_comp.max())
        
        # Update offset for next component
        current_offset += step_range

    # Concatenate all data
    x_unified = np.concatenate(x_unified_list)
    y_unified = np.concatenate(y_unified_list)

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(x_unified, y_unified, linewidth=2, label="Unified Curve")
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("reward_mean", fontsize=12)
    plt.title(f"Mean Reward Curve ({curve_name})", fontsize=14)
    plt.grid(True, alpha=0.3)

    # Mark maximum value on the unified curve
    max_idx = int(np.argmax(y_unified))
    max_x = x_unified[max_idx]
    max_y = y_unified[max_idx]
    plt.scatter(max_x, max_y, color="red", zorder=5, s=100, label="Max value")
    plt.text(
        max_x,
        max_y,
        f" max={max_y:.2f}",
        color="white",
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="red",
            edgecolor="darkred",
            linewidth=2,
            alpha=0.9
        ),
        zorder=6
    )

    # Mark final value on the unified curve
    final_x = x_unified[-1]
    final_y = y_unified[-1]
    plt.scatter(final_x, final_y, color="green", zorder=5, s=100, label="Final value")
    plt.text(
        final_x,
        final_y,
        f" final={final_y:.2f}",
        color="white",
        fontsize=11,
        fontweight="bold",
        ha="right",
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="green",
            edgecolor="darkgreen",
            linewidth=2,
            alpha=0.9
        ),
        zorder=6
    )

    plt.legend()

    # Add vertical lines to separate components (except the last one)
    for i, sep_x in enumerate(x_separators[:-1]):  # Don't add line after last component
        plt.axvline(x=sep_x, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    # plt.show()
    output_dir = os.path.join(project_root, "curves", "images")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{curve_name}_curve.png"))

if __name__ == "__main__":
    main()
    
# python scripts/visualization/unify_curve.py --curve_name "normal_compressibility_50_steps" --csv_path "wandb_export_2026-01-12T13_50_29.052+05_45.csv"
# python scripts/visualization/unify_curve.py --curve_name "incremental_compressibility_30_to_50" --csv_path "wandb_export_2026-01-12T14_19_42.688+05_45.csv"
# python scripts/visualization/unify_curve.py --curve_name "normal_compressibility_30_steps" --csv_path "wandb_export_2026-01-12T14_24_54.954+05_45.csv"