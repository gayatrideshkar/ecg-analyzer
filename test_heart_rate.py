#!/usr/bin/env python3
"""
Test script to verify heart rate detection improvements
"""
import os
import sys
import numpy as np

# Add the project path to sys.path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ecg_app.advanced_ecg_analyzer import AdvancedECGAnalyzer

def create_synthetic_ecg(duration_sec=10, heart_rate=75, sampling_rate=500):
    """Create a synthetic ECG signal with known heart rate"""
    
    # Time vector
    t = np.linspace(0, duration_sec, int(duration_sec * sampling_rate))
    
    # Heart period in seconds
    heart_period = 60.0 / heart_rate
    
    # Create QRS complexes at regular intervals
    signal = np.zeros(len(t))
    
    # Add QRS complexes
    beats_count = 0
    for i in range(int(duration_sec / heart_period) + 2):  # Add extra to ensure we get enough beats
        qrs_time = i * heart_period
        qrs_idx = int(qrs_time * sampling_rate)
        
        if qrs_idx < len(signal) - 50:
            beats_count += 1
            # Create a more realistic QRS complex
            # R wave (tall spike)
            for j in range(-15, 16):
                if 0 <= qrs_idx + j < len(signal):
                    # R wave
                    if abs(j) <= 5:
                        signal[qrs_idx + j] = 15.0 * np.exp(-j**2 / 10)
                    # Q and S waves (small negative deflections)
                    elif j == -8:
                        signal[qrs_idx + j] = -2.0
                    elif j == 8:
                        signal[qrs_idx + j] = -3.0
                        
            # Add T wave (smaller positive wave after QRS)
            t_wave_idx = qrs_idx + int(0.3 * sampling_rate)  # ~300ms after QRS
            if t_wave_idx < len(signal) - 30:
                for j in range(-20, 21):
                    if 0 <= t_wave_idx + j < len(signal):
                        signal[t_wave_idx + j] += 3.0 * np.exp(-j**2 / 100)
    
    # Add baseline drift
    baseline = 2.0 * np.sin(2 * np.pi * 0.1 * t)
    signal += baseline
    
    # Add some noise but keep it reasonable
    noise = np.random.normal(0, 0.2, len(signal))
    signal += noise
    
    print(f"Generated {beats_count} beats for {heart_rate} BPM over {duration_sec}s (expected: {int(heart_rate * duration_sec / 60)})")
    
    return signal, sampling_rate

def test_heart_rate_detection():
    """Test heart rate detection with known signals"""
    
    test_cases = [
        {"hr": 60, "name": "Normal bradycardia"},
        {"hr": 75, "name": "Normal rate"},
        {"hr": 100, "name": "Upper normal"},
        {"hr": 150, "name": "Tachycardia"},
    ]
    
    print("Testing Heart Rate Detection")
    print("=" * 50)
    
    for test_case in test_cases:
        expected_hr = test_case["hr"]
        name = test_case["name"]
        
        print(f"\nTest: {name} (Expected: {expected_hr} BPM)")
        
        # Create synthetic ECG
        signal_data, fs = create_synthetic_ecg(duration_sec=10, heart_rate=expected_hr, sampling_rate=500)
        
        # Test with analyzer
        analyzer = AdvancedECGAnalyzer()
        analyzer.signal_data = signal_data
        analyzer.sampling_rate = fs
        analyzer.actual_sampling_rate = fs  # Set the actual sampling rate
        
        # Apply filtering
        analyzer.filtered_signal = analyzer._apply_advanced_filtering(signal_data)
        
        # Extract features
        features = analyzer.extract_comprehensive_features()
        
        if features:
            calculated_hr = features.heart_rate
            error = abs(calculated_hr - expected_hr)
            error_percent = (error / expected_hr) * 100
            
            print(f"  Calculated: {calculated_hr:.1f} BPM")
            print(f"  Error: {error:.1f} BPM ({error_percent:.1f}%)")
            
            if error_percent < 10:  # Within 10% is acceptable
                print("  ✅ PASS")
            else:
                print("  ❌ FAIL - Error too large")
        else:
            print("  ❌ FAIL - No features extracted")

if __name__ == "__main__":
    test_heart_rate_detection()