from django.core.management.base import BaseCommand
from ecg_app.models import ECGImage
from ecg_app.views import generate_fallback_analysis

class Command(BaseCommand):
    help = 'Add wave morphology data to all ECG records'

    def handle(self, *args, **options):
        # Get all ECG files
        ecg_files = ECGImage.objects.all()
        
        updated_count = 0
        
        for ecg_file in ecg_files:
            self.stdout.write(f"Updating wave morphology for ECG file #{ecg_file.id}...")
            
            # Get current analysis results
            analysis_result = ecg_file.analysis_results or {}
            
            # Check if wave_morphology exists and has complete data
            needs_update = False
            if 'wave_morphology' not in analysis_result:
                needs_update = True
            else:
                wave_morph = analysis_result['wave_morphology']
                if (not wave_morph.get('p_wave') or 
                    not wave_morph.get('qrs_complex') or 
                    not wave_morph.get('t_wave') or 
                    not wave_morph.get('intervals')):
                    needs_update = True
            
            if needs_update:
                # Generate new analysis to get wave morphology data
                new_analysis = generate_fallback_analysis()
                
                # Update the analysis_results with wave morphology
                if isinstance(analysis_result, dict):
                    analysis_result['wave_morphology'] = new_analysis.get('wave_morphology', {})
                else:
                    analysis_result = new_analysis
                
                ecg_file.analysis_results = analysis_result
                ecg_file.save()
                updated_count += 1
                
                self.stdout.write(
                    self.style.SUCCESS(
                        f'Successfully added wave morphology data to ECG #{ecg_file.id}'
                    )
                )
            else:
                self.stdout.write(f'ECG #{ecg_file.id} already has wave morphology data')
        
        self.stdout.write(
            self.style.SUCCESS(f'Successfully updated {updated_count} ECG records with wave morphology data!')
        )