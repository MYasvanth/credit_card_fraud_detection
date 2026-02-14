"""Test all API endpoints on Render deployment."""

import requests
import json

BASE_URL = "https://credit-card-fraud-detection-cvnl.onrender.com"

def test_endpoints():
    print("=" * 70)
    print("TESTING API ENDPOINTS")
    print("=" * 70)
    
    results = []
    
    # 1. Test root endpoint
    print("\n1. Testing GET /")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=10)
        status = "[PASS]" if response.status_code == 200 else "[FAIL]"
        results.append(("GET /", status, response.status_code))
        print(f"   {status} - Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        results.append(("GET /", "[ERROR]", str(e)))
        print(f"   [ERROR]: {e}")
    
    # 2. Test health endpoint
    print("\n2. Testing GET /health")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        status = "[PASS]" if response.status_code == 200 else "[FAIL]"
        results.append(("GET /health", status, response.status_code))
        print(f"   {status} - Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   Model Status: {response.json().get('status')}")
    except Exception as e:
        results.append(("GET /health", "[ERROR]", str(e)))
        print(f"   [ERROR]: {e}")
    
    # 3. Test predict endpoint
    print("\n3. Testing POST /predict")
    try:
        test_data = {
            "features": {
                "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34,
                "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36, "V10": 0.09,
                "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15": 1.47,
                "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
                "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07, "V25": 0.13,
                "V26": -0.19, "V27": 0.13, "V28": -0.02, "Amount": 149.62
            }
        }
        response = requests.post(f"{BASE_URL}/predict", json=test_data, timeout=30)
        status = "[PASS]" if response.status_code == 200 else "[FAIL]"
        results.append(("POST /predict", status, response.status_code))
        print(f"   {status} - Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            fraud_status = "FRAUD" if result['prediction'] == 1 else "NOT FRAUD"
            print(f"   Prediction: {fraud_status}")
            print(f"   Probability: {result['probability']:.2%}")
            print(f"   Risk Score: {result['risk_score']:.1f}/100")
    except Exception as e:
        results.append(("POST /predict", "[ERROR]", str(e)))
        print(f"   [ERROR]: {e}")
    
    # 4. Test batch predict endpoint
    print("\n4. Testing POST /predict/batch")
    try:
        batch_data = {
            "transactions": [
                {
                    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34,
                    "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36, "V10": 0.09,
                    "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15": 1.47,
                    "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
                    "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07, "V25": 0.13,
                    "V26": -0.19, "V27": 0.13, "V28": -0.02, "Amount": 149.62
                }
            ]
        }
        response = requests.post(f"{BASE_URL}/predict/batch", json=batch_data, timeout=30)
        status = "[PASS]" if response.status_code == 200 else "[FAIL]"
        results.append(("POST /predict/batch", status, response.status_code))
        print(f"   {status} - Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   Processed: {len(response.json())} transactions")
    except Exception as e:
        results.append(("POST /predict/batch", "[ERROR]", str(e)))
        print(f"   [ERROR]: {e}")
    
    # 5. Test metrics endpoint
    print("\n5. Testing GET /metrics")
    try:
        response = requests.get(f"{BASE_URL}/metrics", timeout=10)
        status = "[PASS]" if response.status_code == 200 else "[FAIL]"
        results.append(("GET /metrics", status, response.status_code))
        print(f"   {status} - Status: {response.status_code}")
    except Exception as e:
        results.append(("GET /metrics", "[ERROR]", str(e)))
        print(f"   [ERROR]: {e}")
    
    # 6. Test model info endpoint
    print("\n6. Testing GET /model/info")
    try:
        response = requests.get(f"{BASE_URL}/model/info", timeout=10)
        status = "[PASS]" if response.status_code == 200 else "[FAIL]"
        results.append(("GET /model/info", status, response.status_code))
        print(f"   {status} - Status: {response.status_code}")
    except Exception as e:
        results.append(("GET /model/info", "[ERROR]", str(e)))
        print(f"   [ERROR]: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for endpoint, status, code in results:
        print(f"{status:8} {endpoint:25} {code}")
    
    passed = sum(1 for _, s, _ in results if "PASS" in s)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")
    print("=" * 70)

if __name__ == "__main__":
    test_endpoints()
