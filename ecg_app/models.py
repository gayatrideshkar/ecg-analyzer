from django.db import models
from django.contrib.auth.models import User
import json
import os

class ECGImage(models.Model):
    # Patient Details
    patient_name = models.CharField(max_length=255, blank=False, help_text="Patient's full name")
    patient_age = models.PositiveIntegerField(blank=False, help_text="Patient's age in years")
    patient_dob = models.DateField(blank=False, null=True, help_text="Patient's date of birth")
    patient_gender = models.CharField(max_length=10, choices=[('Male', 'Male'), ('Female', 'Female'), ('Other', 'Other')], default='Other', blank=False, help_text="Patient's gender")
    patient_phone = models.CharField(max_length=20, blank=False, help_text="Patient's 10-digit phone number (format: XXXXXXXXXX)")
    
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


class EchoImage(models.Model):
    """Model for storing 2D Echocardiogram video files and analysis results"""
    
    # Patient Details
    patient_name = models.CharField(max_length=255, blank=False, help_text="Patient's full name")
    patient_age = models.PositiveIntegerField(blank=False, help_text="Patient's age in years")
    patient_dob = models.DateField(blank=False, null=True, help_text="Patient's date of birth")
    patient_gender = models.CharField(max_length=10, choices=[('Male', 'Male'), ('Female', 'Female'), ('Other', 'Other')], default='Other', blank=False, help_text="Patient's gender")
    patient_phone = models.CharField(max_length=20, blank=False, help_text="Patient's 10-digit phone number (format: XXXXXXXXXX)")
    
    # Echo File Information
    echo_file = models.FileField(upload_to='echo_uploads/', help_text="Upload echo video file (MP4, AVI, DICOM)")
    uploaded_at = models.DateTimeField(auto_now_add=True)
    analyzed_at = models.DateTimeField(null=True, blank=True)
    
    # Store detailed analysis results from echo analyzer
    analysis_results = models.JSONField(null=True, blank=True)
    
    # File metadata
    file_name = models.CharField(max_length=255, null=True, blank=True)
    file_size = models.IntegerField(null=True, blank=True)  # Size in bytes
    
    # Analysis summary for quick access
    view_classification = models.CharField(max_length=100, null=True, blank=True)  # A4C, A2C, etc.
    ejection_fraction = models.FloatField(null=True, blank=True)
    ef_classification = models.CharField(max_length=50, null=True, blank=True)  # Normal, Borderline, Reduced
    analysis_quality = models.CharField(max_length=50, null=True, blank=True)
    processing_status = models.CharField(max_length=50, default='pending', choices=[
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('error', 'Error'),
    ])
    error_message = models.TextField(null=True, blank=True)
    overall_impression = models.TextField(null=True, blank=True)
    
    class Meta:
        ordering = ['-uploaded_at']
    
    def __str__(self):
        patient_display = self.patient_name or "Unknown Patient"
        age_display = self.patient_age or "Unknown Age"
        ef_display = f"EF: {self.ejection_fraction}%" if self.ejection_fraction else "EF: Pending"
        return f"Echo {self.id} - {patient_display} ({age_display}y) - {ef_display} - {self.uploaded_at.strftime('%Y-%m-%d %H:%M')}"
    
    def get_file_size_display(self):
        """Return human-readable file size"""
        if self.file_size:
            if self.file_size < 1024:
                return f"{self.file_size} B"
            elif self.file_size < 1024 * 1024:
                return f"{self.file_size / 1024:.1f} KB"
            elif self.file_size < 1024 * 1024 * 1024:
                return f"{self.file_size / (1024 * 1024):.1f} MB"
            else:
                return f"{self.file_size / (1024 * 1024 * 1024):.1f} GB"
        return "Unknown"
    
    def save_analysis_results(self, analysis_data):
        """Save analysis results from echo analyzer"""
        self.analysis_results = analysis_data
        self.analyzed_at = timezone.now()
        
        # Extract summary fields for quick access
        if analysis_data and not analysis_data.get('error'):
            self.view_classification = analysis_data.get('view', '')
            self.ejection_fraction = analysis_data.get('ef')
            
            # Classify EF
            if self.ejection_fraction:
                if self.ejection_fraction >= 50:
                    self.ef_classification = 'Normal'
                elif self.ejection_fraction >= 40:
                    self.ef_classification = 'Borderline'
                else:
                    self.ef_classification = 'Reduced'
            
            self.analysis_quality = f"{analysis_data.get('confidence', 0) * 100:.1f}%" if analysis_data.get('confidence') else None
            self.overall_impression = analysis_data.get('impression', 'Echo analysis completed')
            self.processing_status = 'completed'
        else:
            self.processing_status = 'error'
            self.error_message = analysis_data.get('error', 'Unknown error') if analysis_data else 'Analysis failed'
        
        self.save()
    
    def delete_with_file(self):
        """Delete the Echo record and associated video file"""
        if self.echo_file:
            if os.path.isfile(self.echo_file.path):
                os.remove(self.echo_file.path)
        self.delete()


# Create your models here.

class EchoImage(models.Model):
    # Patient Information
    patient_name = models.CharField(max_length=255, blank=False, help_text="Patient's full name")
    patient_age = models.PositiveIntegerField(blank=False, help_text="Patient's age in years")
    patient_dob = models.DateField(blank=False, null=True, help_text="Patient's date of birth")
    patient_gender = models.CharField(max_length=10, choices=[('Male', 'Male'), ('Female', 'Female'), ('Other', 'Other')], default='Other', blank=False, help_text="Patient's gender")
    patient_phone = models.CharField(max_length=20, blank=False, help_text="Patient's 10-digit phone number (format: XXXXXXXXXX)")
    
    # File Information
    echo_file = models.FileField(upload_to='echo_uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    analyzed_at = models.DateTimeField(null=True, blank=True)
    
    # File metadata
    file_name = models.CharField(max_length=255, null=True, blank=True)
    file_size = models.IntegerField(null=True, blank=True)  # Size in bytes
    
    # Analysis Results (stored as JSON)
    analysis_results = models.JSONField(null=True, blank=True)
    
    # Summary fields for display
    ejection_fraction = models.FloatField(null=True, blank=True)
    lv_function = models.CharField(max_length=50, null=True, blank=True)
    wall_motion = models.CharField(max_length=100, null=True, blank=True)
    image_quality = models.CharField(max_length=50, null=True, blank=True)
    overall_impression = models.TextField(null=True, blank=True)
    
    class Meta:
        ordering = ['-uploaded_at']
    
    def __str__(self):
        patient_display = self.patient_name or "Unknown Patient"
        age_display = self.patient_age or "Unknown Age"
        return f"Echo {self.id} - {patient_display} ({age_display}y) - {self.uploaded_at.strftime('%Y-%m-%d %H:%M')}"
    
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
            # Ejection Fraction
            cardiac_function = analysis_data.get('cardiac_function', {})
            if isinstance(cardiac_function, dict):
                self.ejection_fraction = cardiac_function.get('ejection_fraction')
                self.lv_function = cardiac_function.get('lv_function_grade', 'Unknown')
            
            # Wall Motion
            wall_motion_data = analysis_data.get('wall_motion_analysis', {})
            if isinstance(wall_motion_data, dict):
                self.wall_motion = wall_motion_data.get('overall_assessment', 'Unknown')
            
            # Image quality
            quality_data = analysis_data.get('quality_assessment', {})
            if isinstance(quality_data, dict):
                self.image_quality = f"{quality_data.get('overall_quality', 'Unknown')} ({quality_data.get('confidence', 0)}%)"
            
            # Overall impression
            clinical_data = analysis_data.get('clinical_interpretation', {})
            if isinstance(clinical_data, dict):
                self.overall_impression = clinical_data.get('summary', 'Echo analysis completed')
        
        self.save()
    
    def delete_with_file(self):
        """Delete the Echo record and associated video file"""
        if self.echo_file:
            if os.path.isfile(self.echo_file.path):
                os.remove(self.echo_file.path)
        self.delete()


class EchoImage(models.Model):
    """Model for storing echocardiogram video files and analysis results"""
    
    # Patient Details
    patient_name = models.CharField(max_length=255, blank=False, help_text="Patient's full name")
    patient_age = models.PositiveIntegerField(blank=False, help_text="Patient's age in years")
    patient_dob = models.DateField(blank=False, null=True, help_text="Patient's date of birth")
    patient_gender = models.CharField(max_length=10, choices=[('Male', 'Male'), ('Female', 'Female'), ('Other', 'Other')], default='Other', blank=False, help_text="Patient's gender")
    patient_phone = models.CharField(max_length=20, blank=False, help_text="Patient's 10-digit phone number (format: XXXXXXXXXX)")
    
    # Echo File Information
    echo_file = models.FileField(upload_to='echo_uploads/', help_text="Upload echo video file (MP4, AVI, DICOM)")
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    # Store detailed analysis results from echo analyzer
    analysis_results = models.JSONField(null=True, blank=True)
    
    # File metadata
    file_name = models.CharField(max_length=255, null=True, blank=True)
    file_size = models.IntegerField(null=True, blank=True)  # Size in bytes
    
    # Analysis summary for quick access
    view_classification = models.CharField(max_length=100, null=True, blank=True)  # A4C, A2C, etc.
    ejection_fraction = models.FloatField(null=True, blank=True)
    ef_classification = models.CharField(max_length=50, null=True, blank=True)  # Normal, Borderline, Reduced
    analysis_quality = models.CharField(max_length=50, null=True, blank=True)
    processing_status = models.CharField(max_length=50, default='pending', choices=[
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('error', 'Error'),
    ])
    error_message = models.TextField(null=True, blank=True)
    
    class Meta:
        ordering = ['-uploaded_at']
    
    def __str__(self):
        patient_display = self.patient_name or "Unknown Patient"
        age_display = self.patient_age or "Unknown Age"
        ef_display = f"EF: {self.ejection_fraction}%" if self.ejection_fraction else "EF: Pending"
        return f"Echo {self.id} - {patient_display} ({age_display}y) - {ef_display} - {self.uploaded_at.strftime('%Y-%m-%d %H:%M')}"
    
    def get_file_size_display(self):
        """Return human-readable file size"""
        if self.file_size:
            if self.file_size < 1024:
                return f"{self.file_size} B"
            elif self.file_size < 1024 * 1024:
                return f"{self.file_size / 1024:.1f} KB"
            elif self.file_size < 1024 * 1024 * 1024:
                return f"{self.file_size / (1024 * 1024):.1f} MB"
            else:
                return f"{self.file_size / (1024 * 1024 * 1024):.1f} GB"
        return "Unknown"
    
    def save_analysis_results(self, analysis_data):
        """Save analysis results from echo analyzer"""
        self.analysis_results = analysis_data
        
        # Extract summary fields for quick access
        if analysis_data and not analysis_data.get('error'):
            self.view_classification = analysis_data.get('view', '')
            self.ejection_fraction = analysis_data.get('ef')
            
            # Classify EF
            if self.ejection_fraction:
                if self.ejection_fraction >= 50:
                    self.ef_classification = 'Normal'
                elif self.ejection_fraction >= 40:
                    self.ef_classification = 'Borderline'
                else:
                    self.ef_classification = 'Reduced'
            
            self.analysis_quality = analysis_data.get('confidence', 0) * 100 if analysis_data.get('confidence') else None
            self.processing_status = 'completed'
        else:
            self.processing_status = 'error'
            self.error_message = analysis_data.get('error', 'Unknown error') if analysis_data else 'Analysis failed'
        
        self.save()
    
    def delete_with_file(self):
        """Delete the Echo record and associated video file"""
        if self.echo_file:
            if os.path.isfile(self.echo_file.path):
                os.remove(self.echo_file.path)
        self.delete()


# Create your models here.
