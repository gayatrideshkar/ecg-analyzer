#!/usr/bin/env python
"""
Test script to verify 2D Echo analyzer functionality
"""
import os
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')
django.setup()

from ecg_app.models import ECGImage, EchoImage
from ecg_app.views import generate_echo_fallback_analysis

def test_echo_system():
    print("🫀 Testing 2D Echo Analyzer System")
    print("=" * 50)
    
    # Test 1: Check if Echo model exists
    print("✅ Testing EchoImage model...")
    try:
        echo_count = EchoImage.objects.count()
        print(f"   📊 EchoImage model working - {echo_count} records found")
    except Exception as e:
        print(f"   ❌ EchoImage model error: {e}")
        return False
    
    # Test 2: Check analysis function
    print("✅ Testing Echo analysis function...")
    try:
        analysis_result = generate_echo_fallback_analysis()
        print(f"   🧠 Echo analysis function working")
        print(f"   📈 Sample View: {analysis_result['view']}")
        print(f"   💓 Sample EF: {analysis_result['ef']}%")
        print(f"   📊 Sample Grade: {analysis_result['ef_grade']}")
    except Exception as e:
        print(f"   ❌ Echo analysis function error: {e}")
        return False
    
    # Test 3: Check both models work together
    print("✅ Testing both ECG and Echo models...")
    try:
        ecg_count = ECGImage.objects.count()
        echo_count = EchoImage.objects.count()
        total_files = ecg_count + echo_count
        print(f"   📊 ECG files: {ecg_count}")
        print(f"   🫀 Echo files: {echo_count}")
        print(f"   📁 Total medical files: {total_files}")
    except Exception as e:
        print(f"   ❌ Models integration error: {e}")
        return False
    
    print("\n🎉 2D Echo Analyzer System is ready!")
    print("🌐 Available pages:")
    print("   • ECG Analysis: http://127.0.0.1:8003/")
    print("   • Echo Analysis: http://127.0.0.1:8003/echo/")
    print("   • ECG Files: http://127.0.0.1:8003/uploaded-files/")
    print("   • Echo Files: http://127.0.0.1:8003/echo/files/")
    return True

if __name__ == "__main__":
    test_echo_system()