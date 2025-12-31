"""Test if simple service works."""

import requests
import time
import subprocess
import sys

def test_service():
    """Test the simple BentoML service."""
    print("Testing BentoML Service...")
    
    try:
        # Test home page
        response = requests.get("http://localhost:8080/", timeout=2)
        print(f"Home: {response.json()}")
        
        # Test health
        response = requests.get("http://localhost:8080/health", timeout=2)
        print(f"Health: {response.json()}")
        
        return True
    except Exception as e:
        print(f"Service not running: {e}")
        return False

if __name__ == "__main__":
    if test_service():
        print("BentoML service is working!")
    else:
        print("Start service with: python simple_bento.py")