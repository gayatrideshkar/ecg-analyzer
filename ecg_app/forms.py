from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import ECGImage, EchoImage
from datetime import date


class SignUpForm(UserCreationForm):
    """Custom signup form with additional fields"""
    first_name = forms.CharField(
        max_length=30,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your first name',
            'required': True
        })
    )
    last_name = forms.CharField(
        max_length=30,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your last name',
            'required': True
        })
    )
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your email address',
            'required': True
        })
    )
    password1 = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your password',
            'required': True
        })
    )
    password2 = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Confirm your password',
            'required': True
        })
    )

    class Meta:
        model = User
        fields = ('first_name', 'last_name', 'email', 'password1', 'password2')

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(username=email).exists():
            raise forms.ValidationError('This email address is already registered.')
        return email

    def save(self, commit=True):
        user = super().save(commit=False)
        user.username = self.cleaned_data['email']  # Use email as username
        user.email = self.cleaned_data['email']
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        if commit:
            user.save()
        return user


class LoginForm(forms.Form):
    """Custom login form"""
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your email address',
            'required': True
        })
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your password',
            'required': True
        })
    )


class ECGImageForm(forms.ModelForm):
    """Form for uploading ECG images with patient information"""
    
    class Meta:
        model = ECGImage
        fields = ['patient_name', 'patient_age', 'patient_dob', 'patient_phone', 'patient_gender', 'image']
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
            'image': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'image/*',
                'required': True
            })
        }
    
    def clean_image(self):
        image = self.cleaned_data.get('image')
        if not image:
            raise forms.ValidationError('Please select an ECG image file.')
        
        # Check file size (limit to 10MB)
        if image.size > 10 * 1024 * 1024:
            raise forms.ValidationError('Image file size must be less than 10MB.')
        
        # Check if it's a valid image format
        valid_formats = ['jpeg', 'jpg', 'png', 'bmp', 'tiff', 'gif']
        file_extension = image.name.split('.')[-1].lower()
        if file_extension not in valid_formats:
            raise forms.ValidationError(f'Please upload a valid image file. Supported formats: {", ".join(valid_formats)}')
        
        return image
    
    def clean_patient_phone(self):
        phone = self.cleaned_data.get('patient_phone')
        # Remove any non-digit characters
        phone_digits = ''.join(filter(str.isdigit, phone))
        
        if len(phone_digits) != 10:
            raise forms.ValidationError('Please enter a valid 10-digit phone number.')
        
        return phone_digits
    
    def clean_patient_dob(self):
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
    
    def save(self, commit=True):
        """Custom save method to set file metadata"""
        instance = super().save(commit=False)
        
        # Set file metadata if image is provided
        if instance.image:
            instance.file_name = instance.image.name
            instance.file_size = instance.image.size
        
        if commit:
            instance.save()
        return instance


class EchoUploadForm(forms.ModelForm):
    """Form for uploading and analyzing 2D Echocardiogram videos with patient information"""
    
    patient_name = forms.CharField(
        max_length=255,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter patient full name',
            'required': True
        }),
        help_text="Patient's full name"
    )
    
    patient_age = forms.IntegerField(
        min_value=0,
        max_value=150,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter patient age',
            'required': True,
            'min': 0,
            'max': 150
        }),
        help_text="Patient's age in years"
    )
    
    patient_dob = forms.DateField(
        widget=forms.DateInput(attrs={
            'class': 'form-control',
            'type': 'date',
            'required': True
        }),
        help_text="Patient's date of birth"
    )
    
    patient_gender = forms.ChoiceField(
        choices=[('Male', 'Male'), ('Female', 'Female'), ('Other', 'Other')],
        widget=forms.Select(attrs={
            'class': 'form-control',
            'required': True
        }),
        help_text="Patient's gender"
    )
    
    patient_phone = forms.CharField(
        max_length=20,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter Phone number',
            'pattern': '[0-9]{10}',
            'title': 'Please enter a 10-digit phone number',
            'required': True
        }),
        help_text="Patient's 10-digit phone number"
    )
    
    echo_file = forms.FileField(
        widget=forms.FileInput(attrs={
            'class': 'file-input',
            'accept': '.mp4,.avi,.dcm,.dicom',
            'required': True
        }),
        help_text="Upload echo video file (MP4, AVI, DICOM)"
    )
    
    class Meta:
        model = EchoImage
        fields = ['patient_name', 'patient_age', 'patient_dob', 'patient_gender', 'patient_phone', 'echo_file']
    
    def clean_patient_phone(self):
        phone = self.cleaned_data.get('patient_phone')
        # Remove any non-digit characters
        phone_digits = ''.join(filter(str.isdigit, phone))
        
        if len(phone_digits) != 10:
            raise forms.ValidationError('Please enter a valid 10-digit phone number.')
        
        return phone_digits
    
    def clean_patient_dob(self):
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
    
    def clean_echo_file(self):
        echo_file = self.cleaned_data.get('echo_file')
        if echo_file:
            # Check file size (limit to 500MB)
            if echo_file.size > 500 * 1024 * 1024:
                raise forms.ValidationError('File size should be less than 500MB.')
            
            # Check file extension
            file_name = echo_file.name.lower()
            allowed_extensions = ['.mp4', '.avi', '.dcm', '.dicom']
            if not any(file_name.endswith(ext) for ext in allowed_extensions):
                raise forms.ValidationError('Please upload a valid echo video file (MP4, AVI, or DICOM).')
        
        return echo_file
    
    def save(self, commit=True):
        """Custom save method to set file metadata"""
        instance = super().save(commit=False)
        
        # Set file metadata if echo file is provided
        if instance.echo_file:
            instance.file_name = instance.echo_file.name
            instance.file_size = instance.echo_file.size
        
        if commit:
            instance.save()
        return instance