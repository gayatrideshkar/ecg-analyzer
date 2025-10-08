from django.contrib import admin
from .models import ECGImage

# Register your models here.

@admin.register(ECGImage)
class ECGImageAdmin(admin.ModelAdmin):
    list_display = ('patient_name', 'patient_age', 'patient_dob', 'patient_phone', 'uploaded_at', 'heart_rate', 'rhythm_type')
    list_filter = ('uploaded_at', 'rhythm_type', 'image_quality')
    search_fields = ('patient_name', 'patient_phone')
    ordering = ('-uploaded_at',)
    readonly_fields = ('uploaded_at', 'analysis_results')
    
    fieldsets = (
        ('Patient Information', {
            'fields': ('patient_name', 'patient_age', 'patient_dob', 'patient_phone')
        }),
        ('ECG Data', {
            'fields': ('image', 'uploaded_at')
        }),
        ('Analysis Results', {
            'fields': ('analysis_results', 'heart_rate', 'rhythm_type', 'image_quality'),
            'classes': ('collapse',)
        }),
    )
