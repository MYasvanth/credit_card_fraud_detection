#!/usr/bin/env python3
"""
Automated API testing script - runs all endpoints programmatically
"""

import requests
import json
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

def test_endpoint(method: str, endpoint: str, data: Dict[Any, Any] = None) -> Dict[str, Any]:
    """Test a single API endpoint."""
    url = f"{BASE_URL}{endpoint}"
    start_time = time.time()
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        
        response_time = (time.time() - start_time) * 1000
        
        return {
            "endpoint": endpoint,
            "method": method,
            "status_code": response.status_code,
            "response_time_ms": round(response_time, 2),
            "success": response.status_code == 200,
            "response": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
        }
    except Exception as e:
        return {
            "endpoint": endpoint,
            "method": method,
            "success": False,
            "error": str(e)
        }

def run_all_tests():
    """Run all API endpoint tests."""
    print("🚀 Starting API Tests...")
    print("=" * 50)
    
    # Test data
    single_prediction_data = {
        "features": {
            "V1": -1.359807134, "V2": -0.072781173, "V3": 2.536346738,
            "V4": 1.378155224, "V5": -0.338320770, "V6": 0.462387778,
            "V7": 0.239598554, "V8": 0.098697901, "V9": 0.363786969,
            "V10": 0.090794172, "V11": -0.551599533, "V12": -0.617800856,
            "V13": -0.991389847, "V14": -0.311169354, "V15": 1.468176972,
            "V16": -0.470400525, "V17": 0.207971242, "V18": 0.025791513,
            "V19": 0.403992960, "V20": 0.251412098, "V21": -0.018306778,
            "V22": 0.277837576, "V23": -0.110473910, "V24": 0.066928075,
            "V25": 0.128539358, "V26": -0.189114844, "V27": 0.133558377,
            "V28": -0.021053053, "Amount": 149.62, "Time": 0
        }
    }
    
    batch_prediction_data = {
        "transactions": [
            {**single_prediction_data["features"]},
            {**single_prediction_data["features"], "Amount": 89.50, "Time": 300}
        ]
    }
    
    # Test cases
    tests = [
        ("GET", "/health", None),
        ("POST", "/predict", single_prediction_data),
        ("POST", "/predict/batch", batch_prediction_data),
        ("GET", "/model/info", None),
        ("GET", "/metrics", None),
        ("POST", "/model/reload", None),
        ("GET", "/monitoring/drift", None),
        ("POST", "/ab-test/start?model_a=model_v1&model_b=model_v2", None),
        ("GET", "/comparison/results", None)
    ]
    
    results = []
    passed = 0
    failed = 0
    
    for method, endpoint, data in tests:
        print(f"Testing {method} {endpoint}...")
        result = test_endpoint(method, endpoint, data)
        results.append(result)
        
        if result["success"]:
            print(f"✅ PASS - {result['response_time_ms']}ms")
            passed += 1
        else:
            print(f"❌ FAIL - {result.get('error', f'Status: {result.get(\"status_code\")}')}")
            failed += 1
        
        time.sleep(0.5)  # Small delay between requests
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"Total Tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed/len(tests)*100):.1f}%")
    
    # Detailed results
    print("\n📋 DETAILED RESULTS:")
    for result in results:
        status = "✅" if result["success"] else "❌"
        endpoint = result["endpoint"]
        method = result["method"]
        time_ms = result.get("response_time_ms", "N/A")
        print(f"{status} {method} {endpoint} - {time_ms}ms")
    
    return results

if __name__ == "__main__":
    print("Credit Card Fraud Detection API - Automated Testing")
    print("Make sure the FastAPI server is running on http://localhost:8000")
    print()
    
    try:
        results = run_all_tests()
        
        # Save results to file
        with open("api_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: api_test_results.json")
        
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")