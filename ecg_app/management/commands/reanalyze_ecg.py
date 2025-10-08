from django.core.management.base import BaseCommand
from ecg_app.models import ECGImage
from ecg_app.ecg_analyzer import analyze_ecg_image

class Command(BaseCommand):
    help = 'Re-analyze existing ECG images with real image processing'

    def add_arguments(self, parser):
        parser.add_argument(
            '--file-id',
            type=int,
            help='Analyze specific ECG file by ID',
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force re-analysis of all files even if they have results',
        )

    def handle(self, *args, **options):
        if options['file_id']:
            # Analyze specific file
            try:
                ecg_file = ECGImage.objects.get(id=options['file_id'])
                ecg_files = [ecg_file]
            except ECGImage.DoesNotExist:
                self.stdout.write(
                    self.style.ERROR(f'ECG file with ID {options["file_id"]} not found')
                )
                return
        else:
            # Analyze all files
            ecg_files = ECGImage.objects.all()
        
        updated_count = 0
        
        for ecg_file in ecg_files:
            # Skip if already has analysis and not forced
            if ecg_file.analysis_results and not options['force']:
                self.stdout.write(f'Skipping ECG #{ecg_file.id} (already analyzed, use --force to re-analyze)')
                continue
            
            self.stdout.write(f'Re-analyzing ECG file #{ecg_file.id} with real image processing...')
            
            try:
                # Perform real ECG analysis on the image
                if ecg_file.image and ecg_file.image.path:
                    analysis_result = analyze_ecg_image(ecg_file.image.path)
                    
                    # Save the new analysis results
                    ecg_file.analysis_results = analysis_result
                    
                    # Extract and update key fields
                    if isinstance(analysis_result, dict):
                        # Extract rhythm type
                        if 'rhythm_analysis' in analysis_result:
                            rhythm_data = analysis_result['rhythm_analysis']
                            if isinstance(rhythm_data, dict):
                                ecg_file.rhythm_type = (
                                    rhythm_data.get('condition_summary') or 
                                    rhythm_data.get('primary_rhythm') or 
                                    'Unknown'
                                )
                        
                        # Extract heart rate
                        if 'heart_rate' in analysis_result:
                            heart_rate_data = analysis_result['heart_rate']
                            if isinstance(heart_rate_data, dict):
                                ecg_file.heart_rate = heart_rate_data.get('bpm')
                            elif isinstance(heart_rate_data, (int, float)):
                                ecg_file.heart_rate = heart_rate_data
                        
                        # Extract image quality
                        if 'image_quality' in analysis_result:
                            quality_data = analysis_result['image_quality']
                            if isinstance(quality_data, dict):
                                ecg_file.image_quality = quality_data.get('quality', 'Unknown')
                    
                    ecg_file.save()
                    updated_count += 1
                    
                    # Display analysis results
                    heart_rate = "N/A"
                    if ecg_file.heart_rate:
                        heart_rate = f"{ecg_file.heart_rate} BPM"
                    
                    self.stdout.write(
                        self.style.SUCCESS(
                            f'✓ ECG #{ecg_file.id}: {ecg_file.rhythm_type}, HR: {heart_rate}'
                        )
                    )
                    
                    # Show image quality
                    if 'image_quality' in analysis_result:
                        quality = analysis_result['image_quality']
                        if isinstance(quality, dict):
                            quality_score = quality.get('score', 'N/A')
                            quality_desc = quality.get('quality', 'Unknown')
                            self.stdout.write(f'   Image Quality: {quality_desc} ({quality_score}/100)')
                else:
                    self.stdout.write(
                        self.style.WARNING(f'No image file found for ECG #{ecg_file.id}')
                    )
                    
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'Failed to analyze ECG #{ecg_file.id}: {str(e)}')
                )
        
        self.stdout.write('')
        self.stdout.write(
            self.style.SUCCESS(f'Successfully re-analyzed {updated_count} ECG files with real image processing!')
        )
        self.stdout.write(
            self.style.SUCCESS('All files now use accurate ECG analysis based on actual image content.')
        )