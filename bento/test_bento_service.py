"""Test the BentoML-style service."""

import requests
import json

# Test data
test_features = {f"V{i}": 0.1 * i for i in range(1, 29)}
test_features["Amount"] = 150.0

def test_service():
    """Test the running service."""
    base_url = "http://localhost:5000"
    
    # Test health
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health: {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return
    
    # Test prediction
    print("\nTesting prediction endpoint...")
    try:
        response = requests.post(
            f"{base_url}/predict",
            json={"features": test_features}
        )
        print(f"Prediction: {response.json()}")
    except Exception as e:
        print(f"Prediction failed: {e}")
    
    # Test batch prediction
    print("\nTesting batch prediction...")
    try:
        response = requests.post(
            f"{base_url}/batch_predict",
            json={"transactions": [test_features, test_features]}
        )
        print(f"Batch prediction: {response.json()}")
    except Exception as e:
        print(f"Batch prediction failed: {e}")

if __name__ == "__main__":
    test_service()