"""Script to run A/B testing for model comparison."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, Any, Optional, List
from src.utils.model_comparison import compare_models_from_registry
from src.utils.constants import RAW_DATA_PATH, PROCESSED_DATA_PATH
from src.utils.logger import logger

def load_test_data(data_path: str, target_column: str = "Class") -> pd.DataFrame:
    """Load test data for A/B testing."""
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Test data not found at {data_path}")

    data = pd.read_csv(data_path)
    logger.info(f"Loaded test data with shape: {data.shape}")
    return data

def simulate_ab_test(
    model_a_name: str,
    model_b_name: str,
    test_data: pd.DataFrame,
    traffic_split: float = 0.5,
    n_simulations: int = 1000
) -> Dict[str, Any]:
    """
    Simulate A/B test by routing predictions and collecting metrics.

    Args:
        model_a_name: Name of model A
        model_b_name: Name of model B
        test_data: Test dataset
        traffic_split: Fraction of traffic to model B
        n_simulations: Number of prediction simulations

    Returns:
        A/B test simulation results
    """
    logger.info(f"Simulating A/B test: {model_a_name} vs {model_b_name}")

    # Load models
    comparator = compare_models_from_registry(
        [model_a_name, model_b_name],
        test_data,
        target_column="Class"
    )

    # Evaluate models
    results = comparator.evaluate_all_models()

    if len(results) < 2:
        raise ValueError("Could not load both models for A/B testing")

    model_a_results = results.get(f"{model_a_name}_Production") or results.get(f"{model_a_name}_Staging")
    model_b_results = results.get(f"{model_b_name}_Production") or results.get(f"{model_b_name}_Staging")

    if not model_a_results or not model_b_results:
        raise ValueError("Model results not found")

    # Simulate traffic routing
    predictions_log = []

    for i in range(min(n_simulations, len(test_data))):
        # Route to A or B
        route_to_b = np.random.random() < traffic_split
        group = "B" if route_to_b else "A"

        # Get prediction from routed model
        if route_to_b:
            pred = model_b_results["predictions"][i]
            proba = model_b_results["probabilities"][i]
        else:
            pred = model_a_results["predictions"][i]
            proba = model_a_results["probabilities"][i]

        # Log prediction
        predictions_log.append({
            "sample_id": i,
            "group": group,
            "prediction": int(pred),
            "probability": float(proba),
            "true_label": int(test_data.iloc[i]["Class"])
        })

    # Analyze results
    df_log = pd.DataFrame(predictions_log)

    analysis = {
        "test_info": {
            "model_a": model_a_name,
            "model_b": model_b_name,
            "traffic_split": traffic_split,
            "total_predictions": len(predictions_log),
            "group_a_count": len(df_log[df_log["group"] == "A"]),
            "group_b_count": len(df_log[df_log["group"] == "B"])
        },
        "metrics": {},
        "predictions_log": predictions_log
    }

    # Calculate metrics per group
    for group in ["A", "B"]:
        group_data = df_log[df_log["group"] == group]
        if len(group_data) > 0:
            from sklearn.metrics import classification_report
            report = classification_report(
                group_data["true_label"],
                group_data["prediction"],
                output_dict=True,
                zero_division=0
            )

            analysis["metrics"][group] = {
                "accuracy": report["accuracy"],
                "precision": report["1"]["precision"],
                "recall": report["1"]["recall"],
                "f1_score": report["1"]["f1-score"],
                "sample_size": len(group_data)
            }

    # Compare performance
    if "A" in analysis["metrics"] and "B" in analysis["metrics"]:
        f1_a = analysis["metrics"]["A"]["f1_score"]
        f1_b = analysis["metrics"]["B"]["f1_score"]
        improvement = (f1_b - f1_a) / f1_a * 100

        analysis["comparison"] = {
            "f1_improvement": improvement,
            "winner": "B" if f1_b > f1_a else "A",
            "significant": abs(improvement) > 2.0  # Arbitrary threshold
        }

    logger.info(f"A/B test simulation complete. Results: {analysis['comparison']}")

    return analysis

def main():
    parser = argparse.ArgumentParser(description='Run A/B test simulation for fraud detection models')
    parser.add_argument('--model-a', required=True, help='Name of model A in registry')
    parser.add_argument('--model-b', required=True, help='Name of model B in registry')
    parser.add_argument('--test-data', default=str(RAW_DATA_PATH / "creditcard.csv"),
                       help='Path to test data CSV')
    parser.add_argument('--traffic-split', type=float, default=0.5,
                       help='Traffic split to model B (0.5 = 50/50)')
    parser.add_argument('--n-simulations', type=int, default=1000,
                       help='Number of prediction simulations')
    parser.add_argument('--output', help='Output file for results (JSON)')

    args = parser.parse_args()

    try:
        # Load test data
        test_data = load_test_data(args.test_data)

        # Run A/B test simulation
        results = simulate_ab_test(
            args.model_a,
            args.model_b,
            test_data,
            args.traffic_split,
            args.n_simulations
        )

        # Print results
        print("\n" + "="*60)
        print("A/B TEST SIMULATION RESULTS")
        print("="*60)
        print(f"Model A: {args.model_a}")
        print(f"Model B: {args.model_b}")
        print(f"Traffic Split: {args.traffic_split:.1%} to B")
        print(f"Total Predictions: {results['test_info']['total_predictions']}")
        print("-"*60)

        for group in ["A", "B"]:
            if group in results["metrics"]:
                metrics = results["metrics"][group]
                print(f"Group {group}:")
                print(f"  Sample Size: {metrics['sample_size']}")
                print(f"  F1 Score: {metrics['f1_score']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print()

        if "comparison" in results:
            comp = results["comparison"]
            print("COMPARISON:")
            print(f"  F1 Improvement (B vs A): {comp['f1_improvement']:.2f}%")
            print(f"  Winner: Model {comp['winner']}")
            print(f"  Significant Difference: {'Yes' if comp['significant'] else 'No'}")

        print("="*60)

        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {args.output}")

    except Exception as e:
        logger.error(f"A/B test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
