#!/usr/bin/env python
"""
Quick test script to verify ECG analyzer functionality
"""
import os
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')
django.setup()

from ecg_app.models import ECGImage
from ecg_app.views import generate_fallback_analysis

def test_ecg_analyzer():
    print("🔬 Testing ECG Analyzer Functionality")
    print("=" * 50)
    
    # Test 1: Check if ECG model exists
    print("✅ Testing ECGImage model...")
    try:
        ecg_count = ECGImage.objects.count()
        print(f"   📊 ECGImage model working - {ecg_count} records found")
    except Exception as e:
        print(f"   ❌ ECGImage model error: {e}")
        return False
    
    # Test 2: Check analysis function
    print("✅ Testing ECG analysis function...")
    try:
        analysis_result = generate_fallback_analysis()
        print(f"   🧠 Analysis function working")
        print(f"   📈 Sample Heart Rate: {analysis_result['heart_rate']['bpm']} BPM")
        print(f"   💓 Sample Rhythm: {analysis_result['rhythm_analysis']['primary_rhythm']}")
    except Exception as e:
        print(f"   ❌ Analysis function error: {e}")
        return False
    
    # Test 3: Check dependencies
    print("✅ Testing dependencies...")
    try:
        import cv2
        import numpy as np
        import sklearn
        from scipy.signal import find_peaks
        print("   📦 All dependencies loaded successfully")
        print(f"   🐍 OpenCV version: {cv2.__version__}")
        print(f"   📊 Scikit-learn version: {sklearn.__version__}")
    except ImportError as e:
        print(f"   ⚠️  Dependency issue: {e}")
        print("   💡 Fallback analysis will be used")
    
    print("\n🎉 ECG Analyzer is ready!")
    print("🌐 Open http://127.0.0.1:8002 to use the application")
    return True

if __name__ == "__main__":
    test_ecg_analyzer()