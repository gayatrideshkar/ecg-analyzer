from django.core.management.base import BaseCommand
from ecg_app.models import ECGImage
from ecg_app.views import generate_fallback_analysis

class Command(BaseCommand):
    help = 'Update existing ECG records with missing rhythm_type and heart_rate data'

    def handle(self, *args, **options):
        # Find ECG files missing rhythm_type or heart_rate
        ecg_files = ECGImage.objects.filter(
            rhythm_type__isnull=True
        ) | ECGImage.objects.filter(
            heart_rate__isnull=True
        ) | ECGImage.objects.filter(
            rhythm_type__exact=''
        ) | ECGImage.objects.filter(
            heart_rate__exact=''
        )
        
        updated_count = 0
        
        for ecg_file in ecg_files:
            self.stdout.write(f"Updating ECG file #{ecg_file.id}...")
            
            # Generate new analysis if missing
            if not ecg_file.analysis_results or ecg_file.analysis_results == {}:
                analysis_result = generate_fallback_analysis()
                ecg_file.analysis_results = analysis_result
            else:
                analysis_result = ecg_file.analysis_results
            
            # Extract and save rhythm type and heart rate
            if isinstance(analysis_result, dict):
                if 'rhythm_analysis' in analysis_result and (not ecg_file.rhythm_type or ecg_file.rhythm_type == ''):
                    ecg_file.rhythm_type = analysis_result['rhythm_analysis'].get(
                        'condition_summary', 
                        analysis_result['rhythm_analysis'].get('primary_rhythm', 'Unknown')
                    )
                
                if 'heart_rate' in analysis_result and (not ecg_file.heart_rate or ecg_file.heart_rate == ''):
                    heart_rate_data = analysis_result['heart_rate']
                    if isinstance(heart_rate_data, dict):
                        ecg_file.heart_rate = heart_rate_data.get('bpm', None)
                    else:
                        ecg_file.heart_rate = heart_rate_data
                
                if 'image_quality' in analysis_result and (not ecg_file.image_quality or ecg_file.image_quality == ''):
                    quality_data = analysis_result['image_quality']
                    if isinstance(quality_data, dict):
                        ecg_file.image_quality = quality_data.get('quality', 'Unknown')
                    else:
                        ecg_file.image_quality = quality_data
                
                # Check if wave morphology data is missing or incomplete
                if 'wave_morphology' not in analysis_result or not analysis_result['wave_morphology']:
                    # Generate new analysis with complete wave morphology data
                    new_analysis = generate_fallback_analysis()
                    # Merge the new wave morphology data into existing analysis
                    analysis_result['wave_morphology'] = new_analysis.get('wave_morphology', {})
                    ecg_file.analysis_results = analysis_result
            
            ecg_file.save()
            updated_count += 1
            
            self.stdout.write(
                self.style.SUCCESS(
                    f'Successfully updated ECG #{ecg_file.id}: '
                    f'Condition: {ecg_file.rhythm_type}, '
                    f'Heart Rate: {ecg_file.heart_rate} BPM'
                )
            )
        
        self.stdout.write(
            self.style.SUCCESS(f'Successfully updated {updated_count} ECG records!')
        )