from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from .models import ECGImage


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
    class Meta:
        model = ECGImage
        fields = ['patient_name', 'patient_age', 'patient_dob', 'patient_phone', 'image']
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
            'patient_phone': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter phone number (e.g., +1-234-567-8900)',
                'pattern': r'[+]?[0-9\-\s\(\)]{10,20}',
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
        
        # Remove spaces, dashes, and parentheses for validation
        cleaned_phone = ''.join(char for char in phone if char.isdigit() or char == '+')
        if len(cleaned_phone) < 10:
            raise forms.ValidationError('Please enter a valid phone number with at least 10 digits.')
        return phone
    
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
