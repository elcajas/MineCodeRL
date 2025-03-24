import os
import re
import argparse
import subprocess

# Regex patterns
checkpoint_pattern = re.compile(r"im_update_(\d+)\.pth")
log_line_patterns = {
    "detections": re.compile(r"Total detections: (\d+)"),
    "average_acc": re.compile(r"Average logit of detections: ([\d.]+)")
}

# Function to run the evaluation command
def run_evaluation(checkpoint_path, update_number, la, obj, results_path):
    result_dir = os.path.join(results_path, f"im_update_{update_number}")
    os.makedirs(result_dir, exist_ok=True)  # Ensure the directory exists

    command = f"python test_batch.py -b 0.60 -la {la} --save -o {obj} -p {checkpoint_path}"
    print(f"Running: {command}")
    subprocess.run(command, shell=True, check=True)

# Function to summarize logs for a checkpoint
def summarize_logs_for_checkpoint(update_number, obj, results_path, alpha_values):
    summary = []
    checkpoint_result_dir = os.path.join(results_path, f"im_update_{update_number}")

    for la in alpha_values:
        # Find all directories containing ann_{object}_ and _la{la}_
        log_dirs = [
            d for d in os.listdir(checkpoint_result_dir)
            if f"ann_{obj}_" in d and f"_alpha{la}_" in d
        ]
        if not log_dirs:
            summary.append(f"la={la}  detections=N/A  average ACC=N/A")
            continue

        # Summarize each log directory
        for log_dir in log_dirs:
            log_file = os.path.join(checkpoint_result_dir, log_dir, "output.log")
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    lines = f.readlines()
                    detections = int(log_line_patterns["detections"].search(lines[-2]).group(1))
                    average_acc = float(log_line_patterns["average_acc"].search(lines[-1]).group(1))
                    summary.append(f"la={la}  detections={detections}  average ACC={average_acc}")
            else:
                summary.append(f"la={la}  detections=N/A  average ACC=N/A")
    return summary

# Main processing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation and summarize results.")
    parser.add_argument(
        "-p", "--path",
        type=str,
        required=True,
        help="Base path to the checkpoints directory (e.g., results/trainingtype/traintask/checkpoints)."
    )
    parser.add_argument(
        "-o", "--object",
        type=str,
        required=True,
        help="Object to specify in the test.py command."
    )
    args = parser.parse_args()

    # Extract trainingtype and traintask dynamically
    base_checkpoint_path = args.path
    trainingtype, traintask = base_checkpoint_path.split(os.sep)[1:3]
    base_results_path = os.path.join("images", traintask)

    # Ensure results directory exists
    os.makedirs(base_results_path, exist_ok=True)

    output_file_path = os.path.join(base_results_path, f"summary_{args.object}.txt")

    # Gather and sort checkpoints numerically
    checkpoints = [
        (int(match.group(1)), checkpoint)
        for checkpoint in os.listdir(base_checkpoint_path)
        if (match := checkpoint_pattern.match(checkpoint))
    ]
    checkpoints.sort(key=lambda x: x[0])  # Sort by the numeric part

    with open(output_file_path, "w") as output_file:
        for update_number, checkpoint in checkpoints:
            checkpoint_path = os.path.join(base_checkpoint_path, checkpoint)
            alpha_values = [8, 16, 32, 64]

            # Run evaluation for all la values
            for la in alpha_values:
                run_evaluation(checkpoint_path, update_number, la, args.object, base_results_path)

            # Write summary for this checkpoint
            output_file.write(f"\nSummary `for Checkpoint: {checkpoint}\n")
            output_file.write(f"Path: {checkpoint_path}\n")

            checkpoint_summary = summarize_logs_for_checkpoint(update_number, args.object, base_results_path, alpha_values)
            for line in checkpoint_summary:
                output_file.write(line + "\n")

    print(f"Summary written to {output_file_path}")
