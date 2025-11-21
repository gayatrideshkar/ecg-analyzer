from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from .models import ECGImage, EchoImage
import re


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
            raise ValidationError('This email address is already registered.')
        return email

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        if commit:
            user.save()
        return user


class LoginForm(forms.Form):
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your email address'
        })
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your password'
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
    class Meta:
        model = ECGImage
        fields = ['patient_name', 'patient_age', 'patient_dob', 'patient_gender', 'patient_phone', 'image']
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
                'class': 'form-control phone-input',
                'placeholder': 'Enter Phone number',
                'maxlength': '10',
                'pattern': '[0-9]{10}',
                'inputmode': 'numeric',
                'autocomplete': 'tel',
                'required': True,
                'oninput': 'this.value = this.value.replace(/[^0-9]/g, "")'
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
        phone = self.cleaned_data.get('patient_phone')
        if not phone:
            raise forms.ValidationError('Please enter the patient\'s phone number.')
        
        # Remove any non-digit characters
        digits_only = re.sub(r'\D', '', phone)
        
        # Check if we have exactly 10 digits
        if len(digits_only) != 10:
            raise forms.ValidationError('Please enter exactly 10 digits.')
        
        # Check if first digit is valid (2-9)
        if digits_only[0] in '01':
            raise forms.ValidationError('Area code cannot start with 0 or 1.')
        
        # Return plain 10-digit format
        return digits_only

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
    
    class Meta:
        model = EchoImage
        fields = ['patient_name', 'patient_age', 'patient_dob', 'patient_phone', 'patient_gender', 'echo_file']
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
                'class': 'form-control phone-input',
                'placeholder': 'Enter Phone number',
                'maxlength': '10',
                'pattern': '[0-9]{10}',
                'inputmode': 'numeric',
                'autocomplete': 'tel',
                'required': True,
                'oninput': 'this.value = this.value.replace(/[^0-9]/g, "")'
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
            raise forms.ValidationError(f'File type .{file_extension} is not supported. Please upload a video file.')
        
        # Check file size (limit to 100MB)
        max_size = 100 * 1024 * 1024  # 100MB in bytes
        if echo_file.size > max_size:
            raise forms.ValidationError('File size too large. Please upload a file smaller than 100MB.')
        
        return echo_file

    def clean_patient_phone(self):
        phone = self.cleaned_data.get('patient_phone')
        if not phone:
            raise forms.ValidationError('Please enter a phone number.')
        
        # Remove any non-digit characters
        digits_only = re.sub(r'\D', '', phone)
        
        # Check if we have exactly 10 digits
        if len(digits_only) != 10:
            raise forms.ValidationError('Please enter exactly 10 digits.')
        
        # Check if first digit is valid (2-9)
        if digits_only[0] in '01':
            raise forms.ValidationError('Area code cannot start with 0 or 1.')
        
        # Return plain 10-digit format
        return digits_only

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