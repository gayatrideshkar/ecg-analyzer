#!/usr/bin/env python
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')
django.setup()

from ecg_app.models import ECGImage

def check_database():
    print("=== ECG Database Contents ===")
    print(f"Total records: {ECGImage.objects.count()}")
    print("\nRecent ECG uploads:")
    print("-" * 60)
    
    records = ECGImage.objects.all().order_by('-uploaded_at')[:10]
    
    if records:
        for i, ecg in enumerate(records, 1):
            print(f"{i}. Patient: {ecg.patient_name or 'Unknown'}")
            print(f"   Age: {ecg.patient_age or 'N/A'} | DOB: {ecg.patient_dob or 'N/A'}")
            print(f"   Phone: {ecg.patient_phone or 'N/A'}")
            print(f"   Uploaded: {ecg.uploaded_at.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Heart Rate: {ecg.heart_rate or 'N/A'}")
            print(f"   Rhythm: {ecg.rhythm_type or 'N/A'}")
            print(f"   Image Quality: {ecg.image_quality or 'N/A'}")
            print(f"   File: {ecg.image.name if ecg.image else 'N/A'}")
            print("-" * 60)
    else:
        print("No ECG records found.")

if __name__ == "__main__":
    check_database()