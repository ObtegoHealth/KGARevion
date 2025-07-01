#!/usr/bin/env python3
"""
Test script for KGARevion Scoring API
"""

import requests
import json

# API base URL
BASE_URL = "http://127.0.0.1:8000"

def test_get_endpoint():
    """Test the GET /score endpoint"""
    print("Testing GET /score endpoint...")
    
    # Test data
    triple = {
        "head_entity": "ADH1B",
        "relation": "protein_protein", 
        "tail_entity": "KIF15"
    }
    
    # Make GET request with query parameter
    query_param = json.dumps(triple)
    response = requests.get(f"{BASE_URL}/score", params={"query": query_param})
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ GET request successful!")
        print(f"   Result: {result['result']}")
        print(f"   Confidence: {result['confidence']:.4f}")
        print(f"   Triple: {result['triple']}")
    else:
        print(f"❌ GET request failed: {response.status_code}")
        print(f"   Error: {response.text}")
    
    print()

def test_post_endpoint():
    """Test the POST /score endpoint"""
    print("Testing POST /score endpoint...")
    
    # Test data
    triple = {
        "head_entity": "Clathrin",
        "relation": "interacts with",
        "tail_entity": "FAT3 protein"
    }
    
    # Make POST request
    response = requests.post(f"{BASE_URL}/score", json=triple)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ POST request successful!")
        print(f"   Result: {result['result']}")
        print(f"   Confidence: {result['confidence']:.4f}")
        print(f"   Triple: {result['triple']}")
    else:
        print(f"❌ POST request failed: {response.status_code}")
        print(f"   Error: {response.text}")
    
    print()

def test_health_endpoint():
    """Test the health check endpoint"""
    print("Testing GET /health endpoint...")
    
    response = requests.get(f"{BASE_URL}/health")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Health check successful!")
        print(f"   Status: {result['status']}")
        print(f"   Model loaded: {result['model_loaded']}")
    else:
        print(f"❌ Health check failed: {response.status_code}")
        print(f"   Error: {response.text}")
    
    print()

def test_root_endpoint():
    """Test the root endpoint"""
    print("Testing GET / endpoint...")
    
    response = requests.get(f"{BASE_URL}/")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Root endpoint successful!")
        print(f"   Message: {result['message']}")
        print(f"   Available endpoints: {list(result['endpoints'].keys())}")
    else:
        print(f"❌ Root endpoint failed: {response.status_code}")
        print(f"   Error: {response.text}")
    
    print()

def main():
    """Run all tests"""
    print("🧪 Testing KGARevion Scoring API")
    print("=" * 50)
    
    try:
        test_health_endpoint()
        test_root_endpoint()
        test_get_endpoint()
        test_post_endpoint()
        
        # Test multiple triples
        print("Testing multiple triples...")
        test_triples = [
            {"head_entity": "AHR", "relation": "target", "tail_entity": "TG"},
            {"head_entity": "Insulin", "relation": "indication", "tail_entity": "Diabetes"},
            {"head_entity": "Aspirin", "relation": "side effect", "tail_entity": "Bleeding"},
        ]
        
        for i, triple in enumerate(test_triples, 1):
            print(f"Test {i}: {triple}")
            response = requests.post(f"{BASE_URL}/score", json=triple)
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Result: {result['result']} (confidence: {result['confidence']:.4f})")
            else:
                print(f"   ❌ Failed: {response.status_code}")
            print()
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the API server.")
        print("   Make sure the server is running with: python KGARevion.py --api")
    except Exception as e:
        print(f"❌ Error during testing: {e}")

if __name__ == "__main__":
    main() 