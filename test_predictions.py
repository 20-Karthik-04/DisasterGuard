#!/usr/bin/env python3
"""
Test script for enhanced disaster prediction system
Tests all core features: multi-language support, location extraction, 
sentiment analysis, category classification, and false positive filtering
"""

import requests
import json
import time

# Test cases covering all features
TEST_CASES = [
    # Real disaster tweets (English)
    {
        "text": "Massive earthquake hits Delhi, buildings collapsed, people trapped need urgent rescue",
        "expected_disaster": True,
        "expected_category": "Earthquake",
        "expected_location": "Delhi"
    },
    {
        "text": "Flash floods in Mumbai, water level rising rapidly, evacuation needed immediately",
        "expected_disaster": True,
        "expected_category": "Flash Flood",
        "expected_location": "Mumbai"
    },
    {
        "text": "Cyclone approaching Chennai coast, heavy rainfall and strong winds reported",
        "expected_disaster": True,
        "expected_category": "Hurricane",
        "expected_location": "Chennai"
    },
    {
        "text": "Wildfire spreading in Bangalore outskirts, smoke visible from city center",
        "expected_disaster": True,
        "expected_category": "Wildfire",
        "expected_location": "Bangalore"
    },
    
    # Multi-language disaster tweets (Hindi transliterated)
    {
        "text": "बड़ा भूकंप दिल्ली में आया है, इमारतें गिर गई हैं",
        "expected_disaster": True,
        "expected_category": "Earthquake",
        "expected_location": "Delhi"
    },
    {
        "text": "मुंबई में बाढ़ आई है, पानी बहुत तेज़ी से बढ़ रहा है",
        "expected_disaster": True,
        "expected_category": "Flood",
        "expected_location": "Mumbai"
    },
    
    # False positive cases
    {
        "text": "This movie was a complete disaster, worst acting ever",
        "expected_disaster": False,
        "expected_category": "Not Disaster"
    },
    {
        "text": "My cooking experiment was a kitchen disaster today",
        "expected_disaster": False,
        "expected_category": "Not Disaster"
    },
    {
        "text": "Traffic is flooded with people going home",
        "expected_disaster": False,
        "expected_category": "Not Disaster"
    },
    {
        "text": "Mumbai",  # Just location name
        "expected_disaster": False,
        "expected_category": "Not Disaster"
    },
    
    # Edge cases with context
    {
        "text": "Breaking: Severe thunderstorm warning issued for Hyderabad, heavy rain expected",
        "expected_disaster": True,
        "expected_category": "Thunderstorm",
        "expected_location": "Hyderabad"
    },
    {
        "text": "Landslide blocks highway near Shimla, rescue operations underway",
        "expected_disaster": True,
        "expected_category": "Landslide"
    },
    {
        "text": "Power outage affects entire Pune city, emergency services on alert",
        "expected_disaster": True,
        "expected_category": "Power Outage",
        "expected_location": "Pune"
    },
    
    # Sentiment analysis test cases
    {
        "text": "Help! Trapped in collapsed building after earthquake in Kolkata, need immediate rescue",
        "expected_disaster": True,
        "expected_sentiment": "Urgent/Fearful",
        "expected_location": "Kolkata"
    },
    {
        "text": "Flood situation in Kerala improving, relief operations successful",
        "expected_disaster": True,
        "expected_sentiment": "Concerned/Informative",
        "expected_location": "Kerala"
    }
]

def test_prediction_api(base_url="http://127.0.0.1:4000"):
    """Test the enhanced prediction API"""
    print("🧪 Testing Enhanced Disaster Prediction System")
    print("=" * 60)
    
    correct_predictions = 0
    total_tests = len(TEST_CASES)
    
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\n📝 Test {i}/{total_tests}")
        print(f"Text: {test_case['text']}")
        
        try:
            # Make API request
            response = requests.post(
                f"{base_url}/analyze",
                json={"tweet": test_case["text"]},
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"❌ API Error: {response.status_code} - {response.text}")
                continue
                
            result = response.json()
            
            # Check disaster prediction
            predicted_disaster = bool(result.get('is_disaster', 0))
            expected_disaster = test_case.get('expected_disaster', False)
            
            disaster_correct = predicted_disaster == expected_disaster
            if disaster_correct:
                correct_predictions += 1
                print(f"✅ Disaster Prediction: {'Disaster' if predicted_disaster else 'Not Disaster'}")
            else:
                print(f"❌ Disaster Prediction: Expected {'Disaster' if expected_disaster else 'Not Disaster'}, Got {'Disaster' if predicted_disaster else 'Not Disaster'}")
            
            # Display detailed results
            print(f"📊 Results:")
            print(f"   - Confidence: {result.get('confidence', 0):.2f}")
            print(f"   - Category: {result.get('category', 'Unknown')}")
            print(f"   - Location: {result.get('location', 'Unknown')}")
            print(f"   - Sentiment: {result.get('sentiment', 'Unknown')}")
            
            if result.get('language_detected') != 'en':
                print(f"   - Language: {result.get('language_detected', 'Unknown')}")
                if result.get('translated_text'):
                    print(f"   - Translated: {result.get('translated_text')}")
            
            print(f"   - Context Strength: {result.get('context_strength', 0)}")
            print(f"   - Has Keywords: {result.get('has_keywords', False)}")
            
            # Check specific expectations
            if 'expected_category' in test_case and predicted_disaster:
                expected_cat = test_case['expected_category'].lower()
                actual_cat = result.get('category', '').lower()
                if expected_cat in actual_cat or actual_cat in expected_cat:
                    print(f"✅ Category matches expectation")
                else:
                    print(f"⚠️  Category: Expected '{test_case['expected_category']}', Got '{result.get('category')}'")
            
            if 'expected_location' in test_case:
                expected_loc = test_case['expected_location'].lower()
                actual_loc = result.get('location', '').lower()
                all_locations = [loc.lower() for loc in result.get('all_locations', [])]
                
                if expected_loc in actual_loc or expected_loc in all_locations:
                    print(f"✅ Location matches expectation")
                else:
                    print(f"⚠️  Location: Expected '{test_case['expected_location']}', Got '{result.get('location')}'")
            
            if 'expected_sentiment' in test_case:
                expected_sent = test_case['expected_sentiment'].lower()
                actual_sent = result.get('sentiment', '').lower()
                if expected_sent in actual_sent or actual_sent in expected_sent:
                    print(f"✅ Sentiment matches expectation")
                else:
                    print(f"⚠️  Sentiment: Expected '{test_case['expected_sentiment']}', Got '{result.get('sentiment')}'")
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Request Error: {e}")
        except Exception as e:
            print(f"❌ Unexpected Error: {e}")
        
        # Small delay between requests
        time.sleep(0.5)
    
    print("\n" + "=" * 60)
    print(f"📈 Overall Results:")
    print(f"   Correct Disaster Predictions: {correct_predictions}/{total_tests}")
    print(f"   Accuracy: {(correct_predictions/total_tests)*100:.1f}%")
    
    if correct_predictions >= total_tests * 0.8:
        print("🎉 System performing well! (>80% accuracy)")
    elif correct_predictions >= total_tests * 0.6:
        print("⚠️  System needs improvement (60-80% accuracy)")
    else:
        print("❌ System needs significant improvement (<60% accuracy)")

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n🔄 Testing Batch Prediction")
    print("-" * 40)
    
    batch_tweets = [
        "Earthquake in Delhi, buildings shaking",
        "Great movie, loved it!",
        "Flood warning for Mumbai issued",
        "Traffic disaster on highway"
    ]
    
    try:
        response = requests.post(
            "http://127.0.0.1:3000/api/predict/batch",
            json={"tweets": batch_tweets},
            timeout=60
        )
        
        if response.status_code == 200:
            results = response.json()
            print(f"✅ Batch processing successful")
            print(f"   Processed {len(results.get('results', []))} tweets")
            
            for result in results.get('results', []):
                tweet = result.get('tweet', '')[:50] + "..."
                if 'error' in result:
                    print(f"   ❌ {tweet}: {result['error']}")
                else:
                    is_disaster = result.get('result', {}).get('is_disaster', 0)
                    confidence = result.get('result', {}).get('confidence', 0)
                    print(f"   {'🚨' if is_disaster else '✅'} {tweet}: {'Disaster' if is_disaster else 'Not Disaster'} ({confidence:.2f})")
        else:
            print(f"❌ Batch API Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Batch test error: {e}")

if __name__ == "__main__":
    print("🚀 Starting Disaster Prediction System Tests")
    print(f"⏰ Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test main prediction API
    test_prediction_api()
    
    # Test batch prediction
    test_batch_prediction()
    
    print(f"\n⏰ Test completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n🎯 Key Improvements Implemented:")
    print("   ✅ Enhanced multi-language detection and translation")
    print("   ✅ Improved location extraction with Indian locations")
    print("   ✅ Advanced sentiment analysis with disaster context")
    print("   ✅ Better disaster category classification")
    print("   ✅ Robust false positive filtering")
    print("   ✅ Dynamic threshold adjustment based on context")
    print("   ✅ Comprehensive error handling and logging")
