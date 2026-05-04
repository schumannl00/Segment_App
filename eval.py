import os
import json
import argparse
import numpy as np
import SimpleITK as sitk

def compute_dice(pred, gt):
    """Compute the Dice Similarity Coefficient for two binary masks."""
    intersection = np.sum(pred * gt)
    sum_total = np.sum(pred) + np.sum(gt)
    if sum_total == 0:
        return 1.0  # Perfect match if both are empty
    return (2. * intersection) / sum_total

def main():
    parser = argparse.ArgumentParser(
        description="nnU-Net style validation script using Dice Similarity Coefficient."
    )
    
    # Required basic inputs
    parser.add_argument(
        "-gt", "--ground_truth", 
        required=True, 
        type=str, 
        help="Path to the folder containing ground truth segmentations (.nii.gz)"
    )
    parser.add_argument(
        "-pred", "--predictions", 
        required=True, 
        type=str, 
        help="Path to the folder containing predicted segmentations (.nii.gz)"
    )
    
    # Range arguments instead of a long list
    parser.add_argument(
        "-start", "--start_label", 
        required=True, 
        type=int, 
        help="The first foreground label to evaluate (e.g., 1)"
    )
    parser.add_argument(
        "-end", "--end_label", 
        required=True, 
        type=int, 
        help="The last foreground label to evaluate (inclusive, e.g., 3)"
    )
    
    parser.add_argument(
        "-o", "--output", 
        default="summary.json", 
        type=str, 
        help="Output filename or path for the summary json file"
    )

    args = parser.parse_args()

    # Generate the sequence of labels dynamically (inclusive of end_label)
    labels = list(range(args.start_label, args.end_label + 1))

    # Determine final output path
    output_path = args.output
    if not os.path.isabs(output_path) and not output_path.startswith("."):
        output_path = os.path.join(args.predictions, output_path)

    # Replicate exact nnU-Net summary format structure
    results = {"results": {"all": [], "mean": {}}}
    
    # Collect ground truth files
    gt_files = sorted([f for f in os.listdir(args.ground_truth) if f.endswith('.nii.gz')])
    if not gt_files:
        print(f"No .nii.gz files found in ground truth folder: {args.ground_truth}")
        return

    # Track metrics per class for the final global mean
    class_dices = {label: [] for label in labels}

    print(f"Evaluating labels {labels[0]} through {labels[-1]} across {len(gt_files)} test files...")

    for gt_name in gt_files:
        pred_path = os.path.join(args.predictions, gt_name)
        gt_path = os.path.join(args.ground_truth, gt_name)

        if not os.path.exists(pred_path):
            print(f"Warning: Prediction missing for {gt_name}. Skipping...")
            continue

        # Read images
        gt_img = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
        pred_img = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))

        case_metrics = {
            "reference": str(os.path.abspath(gt_path)),
            "test": str(os.path.abspath(pred_path)),
            "metrics": {}
        }

        # Calculate metrics per foreground class
        for label in labels:
            gt_mask = (gt_img == label).astype(np.uint8)
            pred_mask = (pred_img == label).astype(np.uint8)

            dice_score = compute_dice(pred_mask, gt_mask)
            
            # Use native Python float to prevent serialization issues
            case_metrics["metrics"][str(label)] = {
                "Dice": float(dice_score)
            }
            class_dices[label].append(dice_score)

        results["results"]["all"].append(case_metrics)

    # Compute per-class means
    mean_metrics = {}
    for label in labels:
        mean_dice = np.mean(class_dices[label]) if class_dices[label] else 0.0
        mean_metrics[str(label)] = {
            "Dice": float(mean_dice)
        }
    
    # Compute overall mean across all foreground classes
    all_class_means = [m["Dice"] for m in mean_metrics.values()]
    mean_metrics["mean_Dice"] = float(np.mean(all_class_means)) if all_class_means else 0.0

    results["results"]["mean"] = mean_metrics

    # Save to file
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\n" + "="*40)
    print(f"Evaluation complete! Summary saved to: {output_path}")
    print(f"Overall Foreground Mean Dice: {mean_metrics['mean_Dice']:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()