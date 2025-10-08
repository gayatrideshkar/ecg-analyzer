from django.db import models
import json
import os

class ECGImage(models.Model):
    # Patient Details
    patient_name = models.CharField(max_length=255, blank=False, help_text="Patient's full name")
    patient_age = models.PositiveIntegerField(blank=False, help_text="Patient's age in years")
    patient_dob = models.DateField(blank=False, null=True, help_text="Patient's date of birth")
    patient_phone = models.CharField(max_length=20, blank=False, help_text="Patient's phone number")
    
    # File Information
    image = models.ImageField(upload_to='ecg_uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    # Store detailed analysis results
    analysis_results = models.JSONField(null=True, blank=True)
    
    # File metadata
    file_name = models.CharField(max_length=255, null=True, blank=True)
    file_size = models.IntegerField(null=True, blank=True)  # Size in bytes
    
    # Analysis summary for quick access
    heart_rate = models.CharField(max_length=50, null=True, blank=True)
    rhythm_type = models.CharField(max_length=100, null=True, blank=True)
    image_quality = models.CharField(max_length=50, null=True, blank=True)
    overall_impression = models.TextField(null=True, blank=True)
    
    class Meta:
        ordering = ['-uploaded_at']
    
    def __str__(self):
        patient_display = self.patient_name or "Unknown Patient"
        age_display = self.patient_age or "Unknown Age"
        return f"ECG {self.id} - {patient_display} ({age_display}y) - {self.uploaded_at.strftime('%Y-%m-%d %H:%M')}"
    
    def get_file_size_display(self):
        """Return human-readable file size"""
        if self.file_size:
            if self.file_size < 1024:
                return f"{self.file_size} B"
            elif self.file_size < 1024 * 1024:
                return f"{self.file_size / 1024:.1f} KB"
            else:
                return f"{self.file_size / (1024 * 1024):.1f} MB"
        return "Unknown"
    
    def save_analysis_results(self, analysis_data):
        """Save detailed analysis results and summary data"""
        self.analysis_results = analysis_data
        
        # Extract summary data for quick access
        if isinstance(analysis_data, dict) and 'error' not in analysis_data:
            # Heart rate
            heart_rate_data = analysis_data.get('heart_rate', {})
            if isinstance(heart_rate_data, dict) and heart_rate_data.get('bpm') != "Unable to determine":
                self.heart_rate = f"{heart_rate_data.get('bpm')} BPM"
            else:
                self.heart_rate = "Unable to determine"
            
            # Rhythm
            rhythm_data = analysis_data.get('rhythm_analysis', {})
            if isinstance(rhythm_data, dict):
                self.rhythm_type = rhythm_data.get('primary_rhythm', 'Unknown')
            
            # Image quality
            quality_data = analysis_data.get('image_quality', {})
            if isinstance(quality_data, dict):
                self.image_quality = f"{quality_data.get('quality')} ({quality_data.get('score')}%)"
            
            # Overall impression
            clinical_data = analysis_data.get('clinical_findings', {})
            if isinstance(clinical_data, dict):
                self.overall_impression = clinical_data.get('overall_impression', 'Analysis completed')
        
        self.save()
    
    def delete_with_file(self):
        """Delete the ECG record and associated image file"""
        if self.image:
            if os.path.isfile(self.image.path):
                os.remove(self.image.path)
        self.delete()

# Create your models here.
