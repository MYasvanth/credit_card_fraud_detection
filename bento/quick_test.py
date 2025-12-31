"""Quick test of BentoML service."""

import requests
import json

def test_service():
    """Test the service quickly."""
    base_url = "http://localhost:8080"
    
    # Test features
    features = {f"V{i}": 0.1 * i for i in range(1, 29)}
    features["Amount"] = 100.0
    
    print("Testing BentoML Service...")
    
    # Health check
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"Health: {response.json()}")
    except Exception as e:
        print(f"Service not running: {e}")
        return False
    
    # Single prediction
    try:
        response = requests.post(
            f"{base_url}/predict",
            json={"features": features},
            timeout=5
        )
        result = response.json()
        print(f"Prediction: {result['prediction']}")
        print(f"Risk Score: {result['risk_score']:.1f}%")
        print(f"Confidence: {result['confidence']}")
        return True
    except Exception as e:
        print(f"Prediction failed: {e}")
        return False

if __name__ == "__main__":
    test_service()