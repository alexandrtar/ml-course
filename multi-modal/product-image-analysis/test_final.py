#!/usr/bin/env python3
"""
Final test client for the Product Success Prediction API
"""

import requests
import json
import time

API_URL = "http://localhost:8000"

def test_root():
    """Test root endpoint"""
    print("🧪 Testing root endpoint...")
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            print("✅ Root endpoint: Beautiful HTML page loaded")
            return True
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
        return False

def test_health():
    """Test health endpoint"""
    print("\n🧪 Testing health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check: {data}")
            return data.get('model_loaded', False)
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_model_info():
    """Test model info endpoint"""
    print("\n🧪 Testing model info...")
    try:
        response = requests.get(f"{API_URL}/model/info")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model info:")
            print(f"   Type: {data.get('model_type', 'N/A')}")
            print(f"   Features: {data.get('n_features', 'N/A')}")
            print(f"   Feature names: {data.get('feature_count', 'N/A')}")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model info error: {e}")
        return False

def test_metrics():
    """Test metrics endpoint"""
    print("\n🧪 Testing metrics...")
    try:
        response = requests.get(f"{API_URL}/model/metrics")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model metrics:")
            print(f"   ROC-AUC: {data.get('roc_auc', 'N/A')}")
            print(f"   F1-Score: {data.get('f1_score', 'N/A')}")
            print(f"   Accuracy: {data.get('accuracy', 'N/A')}")
            return True
        else:
            print(f"❌ Metrics failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Metrics error: {e}")
        return False

def test_feature_importance():
    """Test feature importance endpoint"""
    print("\n🧪 Testing feature importance...")
    try:
        response = requests.get(f"{API_URL}/features/importance")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Feature importance (top 5):")
            for i, feature in enumerate(data[:5]):
                print(f"   {i+1}. {feature['feature']}: {feature['importance']:.4f}")
            return True
        else:
            print(f"❌ Feature importance failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Feature importance error: {e}")
        return False

def test_prediction():
    """Test prediction endpoint"""
    print("\n🧪 Testing prediction...")
    
    test_product = {
        "product_data": {
            "id": 999,
            "gender": "Men",
            "masterCategory": "Apparel",
            "subCategory": "Topwear",
            "articleType": "Tshirts",
            "baseColour": "Black",
            "season": "Summer",
            "year": 2023,
            "usage": "Casual",
            "productDisplayName": "Test T-Shirt"
        },
        "image_url": ""
    }
    
    try:
        response = requests.post(f"{API_URL}/predict", json=test_product)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Prediction successful:")
            print(f"   Product ID: {data['product_id']}")
            print(f"   Success Probability: {data['success_probability']:.3f}")
            print(f"   Prediction: {'Success' if data['prediction'] == 1 else 'Needs Improvement'}")
            print(f"   Confidence: {data['confidence']:.3f}")
            print(f"   Message: {data['message']}")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting FINAL API tests...")
    print("=" * 60)
    
    # Wait a bit for API to start
    print("⏳ Waiting for API to start...")
    time.sleep(3)
    
    tests = [
        test_root,
        test_health,
        test_model_info,
        test_metrics,
        test_feature_importance,
        test_prediction
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            time.sleep(1)  # Пауза между тестами
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("📊 FINAL TEST SUMMARY:")
    passed = sum(results)
    total = len(results)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Your API is fully operational! 🎉")
        print(f"\n🌐 Next steps:")
        print(f"   1. Visit http://localhost:8000/ for the main page")
        print(f"   2. Visit http://localhost:8000/docs for interactive API documentation")
        print(f"   3. Start integrating the API into your applications!")
    else:
        print("⚠️ Some tests failed. Please check the setup.")
    
    print(f"\n💡 Tip: The API is now ready for production use!")

if __name__ == "__main__":
    main()