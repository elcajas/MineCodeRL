import os
import re
import argparse
import matplotlib.pyplot as plt
import pandas as pd

# Baseline values for each object
BASELINES = {
    "spider": {"detections": 51, "acc": 0.74},
    "zombie": {"detections": 20, "acc": 0.66},
}

# Regex patterns
checkpoint_pattern = re.compile(r"im_update_(\d+)\.pth")
la_pattern = re.compile(r"la=(\d+)\s+detections=(\d+)\s+average ACC=([\d.]+)")

# Function to parse the summary file
def parse_summary(file_path):
    data = []
    with open(file_path, "r") as f:
        lines = f.readlines()

    current_checkpoint = None
    for line in lines:
        # Check for a checkpoint line
        checkpoint_match = checkpoint_pattern.search(line)
        if checkpoint_match:
            current_checkpoint = int(checkpoint_match.group(1))

        # Check for a la line
        la_match = la_pattern.search(line)
        if la_match and current_checkpoint is not None:
            la_value = int(la_match.group(1))
            detections = int(la_match.group(2))
            avg_acc = float(la_match.group(3))
            data.append({
                "checkpoint": current_checkpoint,
                "la": la_value,
                "detections": detections,
                "avg_acc": avg_acc
            })

    return pd.DataFrame(data)

# Function to create a grid plot for detections and accuracies
def create_grid_plot(df, object_name, output_file):
    # Sort data by checkpoint
    df = df.sort_values(by=["checkpoint", "la"])

    # Get unique alphas and checkpoints
    alphas = df["la"].unique()
    checkpoints = sorted(df["checkpoint"].unique())
    checkpoint_indices = range(len(checkpoints))

    # Map checkpoints to evenly spaced indices
    checkpoint_mapping = {cp: idx for idx, cp in enumerate(checkpoints)}
    df["checkpoint_index"] = df["checkpoint"].map(checkpoint_mapping)

    # Determine y-axis limits
    y_lim_detections = [0, max(df["detections"].max(), BASELINES[object_name]["detections"] + 10)]
    y_lim_acc = [0, max(df["avg_acc"].max(), BASELINES[object_name]["acc"] + 0.1)]

    # Create the grid plot
    fig, axes = plt.subplots(2, len(alphas), figsize=(5 * len(alphas), 10))

    for col, alpha in enumerate(alphas):
        alpha_data = df[df["la"] == alpha]

        # Plot detections in the first row
        axes[0, col].plot(alpha_data["checkpoint_index"], alpha_data["detections"], marker='o', label=f"la={alpha}")
        axes[0, col].axhline(y=BASELINES[object_name]["detections"], color='red', linestyle='--', label="Baseline")
        axes[0, col].set_title(f"Detections (Alpha={alpha})")
        axes[0, col].set_xticks(checkpoint_indices)
        axes[0, col].set_xticklabels(checkpoints, rotation=75)  # Rotate labels
        axes[0, col].set_xlabel("Checkpoint")
        axes[0, col].set_ylabel("Detections")
        axes[0, col].set_ylim(y_lim_detections)
        axes[0, col].grid()
        axes[0, col].legend()

        # Plot accuracies in the second row
        axes[1, col].plot(alpha_data["checkpoint_index"], alpha_data["avg_acc"], marker='o', label=f"la={alpha}")
        axes[1, col].axhline(y=BASELINES[object_name]["acc"], color='red', linestyle='--', label="Baseline")
        axes[1, col].set_title(f"Accuracy (Alpha={alpha})")
        axes[1, col].set_xticks(checkpoint_indices)
        axes[1, col].set_xticklabels(checkpoints, rotation=75)  # Rotate labels
        axes[1, col].set_xlabel("Checkpoint")
        axes[1, col].set_ylabel("Accuracy")
        axes[1, col].set_ylim(y_lim_acc)
        axes[1, col].grid()
        axes[1, col].legend()

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Grid plot saved to {output_file}")
    plt.close()


# Main script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process summary data and create grid plot with baselines.")
    parser.add_argument(
        "-s", "--summary",
        type=str,
        required=True,
        help="Path to the summary_{object}.txt file."
    )
    args = parser.parse_args()

    # Get the object name from the file name
    summary_file_path = args.summary
    object_name = os.path.basename(summary_file_path).split("_")[1].split(".")[0]
    if object_name not in BASELINES:
        raise ValueError(f"Baseline values for object '{object_name}' are not defined.")

    # Output path
    output_dir = os.path.join(os.path.dirname(summary_file_path), "plots")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"grid_plot_{object_name}.png")

    # Parse the summary file
    df = parse_summary(summary_file_path)

    # Check if data is valid
    if not df.empty:
        print("Parsed Data:")
        print(df)

        # Create grid plot
        create_grid_plot(df, object_name, output_file)
    else:
        print("No data found in the summary file.")
