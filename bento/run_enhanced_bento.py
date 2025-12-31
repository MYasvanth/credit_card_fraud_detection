"""Run enhanced BentoML service with monitoring."""

import subprocess
import time
import requests
import json
from pathlib import Path

def test_enhanced_service():
    """Test enhanced BentoML service."""
    base_url = "http://localhost:5000"
    
    # Test data
    test_features = {f"V{i}": 0.1 * i for i in range(1, 29)}
    test_features["Amount"] = 150.0
    
    print("🧪 Testing Enhanced BentoML Service...")
    
    # Test health with metrics
    print("\n📊 Health Check:")
    try:
        response = requests.get(f"{base_url}/health")
        health_data = response.json()
        print(f"Status: {health_data['status']}")
        print(f"Success Rate: {health_data['metrics']['success_rate']:.2%}")
        print(f"Avg Response Time: {health_data['metrics']['avg_response_time_ms']:.2f}ms")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test enhanced prediction
    print("\n🔮 Enhanced Prediction:")
    try:
        response = requests.post(
            f"{base_url}/predict",
            json={"features": test_features}
        )
        pred_data = response.json()
        print(f"Prediction: {pred_data['prediction']}")
        print(f"Risk Score: {pred_data['risk_score']:.1f}%")
        print(f"Confidence: {pred_data['confidence']}")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
    
    # Test batch prediction
    print("\n📦 Batch Prediction:")
    try:
        response = requests.post(
            f"{base_url}/batch_predict",
            json={"transactions": [test_features] * 3}
        )
        batch_data = response.json()
        print(f"Processed: {batch_data['count']} transactions")
        print(f"Average Risk: {sum(p['risk_score'] for p in batch_data['predictions'])/len(batch_data['predictions']):.1f}%")
    except Exception as e:
        print(f"❌ Batch prediction failed: {e}")
    
    # Test metrics endpoint
    print("\n📈 Service Metrics:")
    try:
        response = requests.get(f"{base_url}/metrics")
        metrics = response.json()
        print(f"Total Requests: {metrics.get('requests_total', 0)}")
        print(f"Success Requests: {metrics.get('requests_success', 0)}")
        print(f"Error Requests: {metrics.get('requests_error', 0)}")
    except Exception as e:
        print(f"❌ Metrics failed: {e}")

def run_load_test():
    """Simple load test."""
    print("\n🚀 Running Load Test...")
    
    test_features = {f"V{i}": 0.1 * i for i in range(1, 29)}
    test_features["Amount"] = 150.0
    
    start_time = time.time()
    success_count = 0
    
    for i in range(50):
        try:
            response = requests.post(
                "http://localhost:5000/predict",
                json={"features": test_features},
                timeout=5
            )
            if response.status_code == 200:
                success_count += 1
        except:
            pass
    
    duration = time.time() - start_time
    print(f"✅ Load Test Results:")
    print(f"   Requests: 50")
    print(f"   Success: {success_count}")
    print(f"   Duration: {duration:.2f}s")
    print(f"   RPS: {50/duration:.1f}")

if __name__ == "__main__":
    print("🎯 Enhanced BentoML Service Tester")
    
    # Wait for service to start
    print("⏳ Waiting for service to start...")
    time.sleep(2)
    
    # Run tests
    test_enhanced_service()
    run_load_test()
    
    print("\n✨ Testing completed!")