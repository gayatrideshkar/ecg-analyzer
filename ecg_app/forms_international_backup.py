from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from .models import ECGImage, EchoImage


class SignUpForm(UserCreationForm):
    first_name = forms.CharField(
        max_length=30,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your first name'
        })
    )
    last_name = forms.CharField(
        max_length=30,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your last name'
        })
    )
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your email address'
        })
    )
    password1 = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Create a password'
        })
    )
    password2 = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Confirm your password'
        })
    )

    class Meta:
        model = User
        fields = ('first_name', 'last_name', 'email', 'password1', 'password2')

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise ValidationError("An account with this email already exists.")
        return email

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        user.username = self.cleaned_data['email']  # Use email as username
        if commit:
            user.save()
        return user


class LoginForm(forms.Form):
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your email address',
            'autocomplete': 'new-email'
        })
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your password',
            'autocomplete': 'new-password'
        })
    )
    
    def clean_email(self):
        email = self.cleaned_data.get('email')
        if not email:
            raise forms.ValidationError('Please enter your email address.')
        return email
    
    def clean_password(self):
        password = self.cleaned_data.get('password')
        if not password:
            raise forms.ValidationError('Please enter your password.')
        if len(password) < 3:
            raise forms.ValidationError('Password is too short.')
        return password

class ECGImageForm(forms.ModelForm):
    COUNTRY_CODES = [
        ('+1', '+1 (US/Canada)'),
        ('+44', '+44 (UK)'),
        ('+33', '+33 (France)'),
        ('+49', '+49 (Germany)'),
        ('+39', '+39 (Italy)'),
        ('+34', '+34 (Spain)'),
        ('+91', '+91 (India)'),
        ('+86', '+86 (China)'),
        ('+81', '+81 (Japan)'),
        ('+7', '+7 (Russia)'),
        ('+55', '+55 (Brazil)'),
        ('+52', '+52 (Mexico)'),
        ('+61', '+61 (Australia)'),
        ('+27', '+27 (South Africa)'),
        ('+20', '+20 (Egypt)'),
        ('+971', '+971 (UAE)'),
        ('+966', '+966 (Saudi Arabia)'),
        ('+82', '+82 (South Korea)'),
        ('+65', '+65 (Singapore)'),
        ('+60', '+60 (Malaysia)'),
    ]
    
    patient_country_code = forms.ChoiceField(
        choices=COUNTRY_CODES,
        initial='+91',
        widget=forms.Select(attrs={
            'class': 'form-control',
            'style': 'width: 140px; display: inline-block; margin-right: 10px;'
        })
    )
    
    class Meta:
        model = ECGImage
        fields = ['patient_name', 'patient_age', 'patient_dob', 'patient_gender', 'patient_country_code', 'patient_phone', 'image']
        widgets = {
            'patient_name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter patient\'s full name',
                'required': True
            }),
            'patient_age': forms.NumberInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter age in years',
                'min': 1,
                'max': 150,
                'required': True
            }),
            'patient_dob': forms.DateInput(attrs={
                'class': 'form-control',
                'placeholder': 'Select date of birth',
                'type': 'date',
                'required': True
            }),
            'patient_gender': forms.Select(attrs={
                'class': 'form-control',
                'required': True
            }),
            'patient_phone': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter phone number (e.g., 9876543210 for India)',
                'style': 'display: inline-block; width: calc(100% - 150px);',
                'maxlength': '25',  # Allow for formatting characters
                'required': True
            }),
            'image': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'image/*,.pdf',
                'required': True
            })
        }
        labels = {
            'patient_name': 'Full Name',
            'patient_age': 'Age',
            'patient_dob': 'Date of Birth',
            'patient_gender': 'Gender',
            'patient_country_code': 'Country Code',
            'patient_phone': 'Phone Number',
            'image': 'ECG Image File'
        }
    
    def clean_patient_age(self):
        age = self.cleaned_data.get('patient_age')
        if age is None:
            raise forms.ValidationError('Please enter the patient\'s age.')
        if age < 1 or age > 150:
            raise forms.ValidationError('Please enter a valid age between 1 and 150 years.')
        return age
    
    def clean_patient_phone(self):
        import re
        phone = self.cleaned_data.get('patient_phone')
        country_code = self.cleaned_data.get('patient_country_code', '+91')
        
        if not phone:
            raise forms.ValidationError('Please enter the patient\'s phone number.')
        
        # Remove brackets, hyphens, spaces, and all non-digit characters
        digits_only = re.sub(r'[()\-\s\D]', '', phone)
        
        # Phone number validation rules by country
        validation_rules = {
            '+1': {'length': 10, 'pattern': r'^[2-9][0-9]{9}$', 'name': 'US/Canada'},
            '+44': {'length': 10, 'pattern': r'^[1-9][0-9]{9}$', 'name': 'UK'},
            '+33': {'length': 9, 'pattern': r'^[1-9][0-9]{8}$', 'name': 'France'},
            '+49': {'length': 11, 'pattern': r'^[1-9][0-9]{10}$', 'name': 'Germany'},
            '+91': {'length': 10, 'pattern': r'^[6-9][0-9]{9}$', 'name': 'India'},
            '+86': {'length': 11, 'pattern': r'^1[3-9][0-9]{9}$', 'name': 'China'},
        }
        
        # Get validation rule for country or use default
        if country_code in validation_rules:
            rule = validation_rules[country_code]
            
            # Validate length
            if len(digits_only) != rule['length']:
                raise forms.ValidationError(f'Please enter a valid {rule["name"]} phone number ({rule["length"]} digits).')
            
            # Validate pattern
            if not re.match(rule['pattern'], digits_only):
                if country_code == '+1':
                    raise forms.ValidationError('US/Canada area code cannot start with 0 or 1.')
                else:
                    raise forms.ValidationError(f'Please enter a valid {rule["name"]} phone number.')
            
            # Format phone number
            if country_code == '+1':
                formatted_phone = f'{country_code} ({digits_only[:3]}) {digits_only[3:6]}-{digits_only[6:]}'
            else:
                formatted_phone = f'{country_code} {digits_only}'
        else:
            # Generic validation for other countries
            if len(digits_only) < 7 or len(digits_only) > 15:
                raise forms.ValidationError('Please enter a valid phone number (7-15 digits).')
            formatted_phone = f'{country_code} {digits_only}'
        
        return formatted_phone
    
    def clean_patient_name(self):
        name = self.cleaned_data.get('patient_name')
        if not name or not name.strip():
            raise forms.ValidationError('Please enter the patient\'s full name.')
        return name.strip()
    
    def clean_patient_dob(self):
        from datetime import date
        dob = self.cleaned_data.get('patient_dob')
        if not dob:
            raise forms.ValidationError('Please enter the patient\'s date of birth.')
        
        # Check if date is not in the future
        if dob > date.today():
            raise forms.ValidationError('Date of birth cannot be in the future.')
        
        # Check if date is reasonable (not more than 150 years ago)
        from datetime import timedelta
        max_age_date = date.today() - timedelta(days=150*365)
        if dob < max_age_date:
            raise forms.ValidationError('Please enter a valid date of birth.')
        
        return dob


class EchoUploadForm(forms.ModelForm):
    """Form for uploading 2D Echocardiogram videos"""
    
    COUNTRY_CODES = [
        ('+1', '+1 (US/Canada)'),
        ('+44', '+44 (UK)'),
        ('+33', '+33 (France)'),
        ('+49', '+49 (Germany)'),
        ('+39', '+39 (Italy)'),
        ('+34', '+34 (Spain)'),
        ('+91', '+91 (India)'),
        ('+86', '+86 (China)'),
        ('+81', '+81 (Japan)'),
        ('+7', '+7 (Russia)'),
        ('+55', '+55 (Brazil)'),
        ('+52', '+52 (Mexico)'),
        ('+61', '+61 (Australia)'),
        ('+27', '+27 (South Africa)'),
        ('+20', '+20 (Egypt)'),
        ('+971', '+971 (UAE)'),
        ('+966', '+966 (Saudi Arabia)'),
        ('+82', '+82 (South Korea)'),
        ('+65', '+65 (Singapore)'),
        ('+60', '+60 (Malaysia)'),
    ]
    
    patient_country_code = forms.ChoiceField(
        choices=COUNTRY_CODES,
        initial='+91',
        widget=forms.Select(attrs={
            'class': 'form-control',
            'style': 'width: 140px; display: inline-block; margin-right: 10px;'
        })
    )
    
    class Meta:
        model = EchoImage
        fields = ['patient_name', 'patient_age', 'patient_dob', 'patient_country_code', 'patient_phone', 'patient_gender', 'echo_file']
        widgets = {
            'patient_name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter patient full name',
                'required': True
            }),
            'patient_age': forms.NumberInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter patient age',
                'min': 0,
                'max': 150,
                'required': True
            }),
            'patient_dob': forms.DateInput(attrs={
                'class': 'form-control',
                'type': 'date',
                'required': True
            }),
            'patient_phone': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter phone number (e.g., 9876543210 for India)',
                'style': 'display: inline-block; width: calc(100% - 150px);',
                'maxlength': '25',  # Allow for formatting characters
                'required': True
            }),
            'patient_gender': forms.Select(attrs={
                'class': 'form-control',
                'required': True
            }),
            'echo_file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'video/*,.avi,.mp4,.mov,.wmv,.flv,.webm',
                'required': True
            })
        }
    
    def clean_echo_file(self):
        echo_file = self.cleaned_data.get('echo_file')
        if not echo_file:
            raise forms.ValidationError('Please select an echocardiogram video file.')
        
        # Check file extension
        allowed_extensions = ['.avi', '.mp4', '.mov', '.wmv', '.flv', '.webm', '.mkv']
        file_extension = echo_file.name.lower().split('.')[-1]
        if f'.{file_extension}' not in allowed_extensions:
            raise forms.ValidationError(
                f'Please upload a valid video file. Supported formats: {", ".join(allowed_extensions)}'
            )
        
        # Check file size (limit to 500MB)
        max_size = 500 * 1024 * 1024  # 500MB in bytes
        if echo_file.size > max_size:
            raise forms.ValidationError(
                f'File size too large. Please upload a file smaller than 500MB. '
                f'Current size: {echo_file.size / (1024*1024):.1f}MB'
            )
        
        return echo_file
    
    def clean_patient_phone(self):
        import re
        phone = self.cleaned_data.get('patient_phone')
        country_code = self.cleaned_data.get('patient_country_code', '+91')
        
        if not phone:
            raise forms.ValidationError('Please enter a phone number.')
        
        # Remove brackets, hyphens, spaces, and all non-digit characters
        digits_only = re.sub(r'[()\-\s\D]', '', phone)
        
        # Phone number validation rules by country
        validation_rules = {
            '+1': {'length': 10, 'pattern': r'^[2-9][0-9]{9}$', 'name': 'US/Canada'},
            '+44': {'length': 10, 'pattern': r'^[1-9][0-9]{9}$', 'name': 'UK'},
            '+33': {'length': 9, 'pattern': r'^[1-9][0-9]{8}$', 'name': 'France'},
            '+49': {'length': 11, 'pattern': r'^[1-9][0-9]{10}$', 'name': 'Germany'},
            '+91': {'length': 10, 'pattern': r'^[6-9][0-9]{9}$', 'name': 'India'},
            '+86': {'length': 11, 'pattern': r'^1[3-9][0-9]{9}$', 'name': 'China'},
        }
        
        # Get validation rule for country or use default
        if country_code in validation_rules:
            rule = validation_rules[country_code]
            
            # Validate length
            if len(digits_only) != rule['length']:
                raise forms.ValidationError(f'Please enter a valid {rule["name"]} phone number ({rule["length"]} digits).')
            
            # Validate pattern
            if not re.match(rule['pattern'], digits_only):
                if country_code == '+1':
                    raise forms.ValidationError('US/Canada area code cannot start with 0 or 1.')
                else:
                    raise forms.ValidationError(f'Please enter a valid {rule["name"]} phone number.')
            
            # Format phone number
            if country_code == '+1':
                formatted_phone = f'{country_code} ({digits_only[:3]}) {digits_only[3:6]}-{digits_only[6:]}'
            else:
                formatted_phone = f'{country_code} {digits_only}'
        else:
            # Generic validation for other countries
            if len(digits_only) < 7 or len(digits_only) > 15:
                raise forms.ValidationError('Please enter a valid phone number (7-15 digits).')
            formatted_phone = f'{country_code} {digits_only}'
        
        return formatted_phone
    
    def clean_patient_name(self):
        name = self.cleaned_data.get('patient_name')
        if not name or not name.strip():
            raise forms.ValidationError('Please enter the patient\'s full name.')
        return name.strip()
    
    def clean_patient_dob(self):
        from datetime import date
        dob = self.cleaned_data.get('patient_dob')
        if not dob:
            raise forms.ValidationError('Please enter the patient\'s date of birth.')
        
        # Check if date is not in the future
        if dob > date.today():
            raise forms.ValidationError('Date of birth cannot be in the future.')
        
        # Check if date is reasonable (not more than 150 years ago)
        from datetime import timedelta
        max_age_date = date.today() - timedelta(days=150*365)
        if dob < max_age_date:
            raise forms.ValidationError('Please enter a valid date of birth.')
        
        return dob
