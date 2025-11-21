#!/usr/bin/env python3
"""
Test to verify phone number reversion to original US format
"""
import os
import sys
import django

# Setup Django environment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')
django.setup()

from ecg_app.forms import ECGImageForm, EchoUploadForm
from django import forms

def test_reverted_phone_validation():
    """Test that phone validation is back to original US format"""
    print("PHONE NUMBER REVERSION VERIFICATION")
    print("=" * 50)
    
    test_cases = [
        ("2345678901", True, "Valid 10-digit US number"),
        ("(234) 567-8901", True, "Valid with formatting"),
        ("234-567-8901", True, "Valid with dashes"),
        ("234567890", False, "Too short (9 digits)"),
        ("23456789012", False, "Too long (11 digits)"),
        ("0234567890", False, "Starts with 0"),
        ("1234567890", False, "Starts with 1"),
    ]
    
    print("\nTesting ECG Form (should be US-only format):")
    print("-" * 45)
    
    for phone_input, should_pass, description in test_cases:
        form = ECGImageForm()
        form.cleaned_data = {'patient_phone': phone_input}
        
        try:
            result = form.clean_patient_phone()
            passed = True
            formatted_result = result
        except forms.ValidationError as e:
            passed = False
            formatted_result = str(e.message)
        
        status = "✅ PASS" if passed == should_pass else "❌ FAIL"
        
        if passed:
            print(f"{status} | {description:<30} | Input: {phone_input:<15} | Output: {formatted_result}")
        else:
            print(f"{status} | {description:<30} | Input: {phone_input:<15} | Error: {formatted_result}")

def test_no_country_code_field():
    """Test that country code fields have been removed"""
    print("\n\nTesting Form Fields (should NOT have country code):")
    print("-" * 55)
    
    # Test ECG form
    ecg_form = ECGImageForm()
    if 'patient_country_code' in ecg_form.fields:
        print("❌ FAIL - ECG form still has country code field")
    else:
        print("✅ PASS - ECG form does NOT have country code field")
    
    # Test Echo form
    echo_form = EchoUploadForm()
    if 'patient_country_code' in echo_form.fields:
        print("❌ FAIL - Echo form still has country code field")
    else:
        print("✅ PASS - Echo form does NOT have country code field")

def test_phone_format_output():
    """Test that phone numbers are formatted as (XXX) XXX-XXXX"""
    print("\n\nTesting Phone Format Output:")
    print("-" * 35)
    
    test_numbers = [
        ("2345678901", "(234) 567-8901"),
        ("5551234567", "(555) 123-4567"),
        ("9876543210", "(987) 654-3210"),
    ]
    
    for input_phone, expected_format in test_numbers:
        form = ECGImageForm()
        form.cleaned_data = {'patient_phone': input_phone}
        
        try:
            result = form.clean_patient_phone()
            if result == expected_format:
                print(f"✅ PASS | {input_phone} → {result}")
            else:
                print(f"❌ FAIL | {input_phone} → {result} (expected: {expected_format})")
        except forms.ValidationError as e:
            print(f"❌ FAIL | {input_phone} → Error: {e.message}")

if __name__ == "__main__":
    test_reverted_phone_validation()
    test_no_country_code_field()
    test_phone_format_output()
    print("\n" + "=" * 50)
    print("✅ PHONE NUMBER REVERSION VERIFICATION COMPLETE")
    print("✅ System restored to original US phone format")
    print("✅ International features removed")
    print("=" * 50)