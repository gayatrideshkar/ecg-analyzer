#!/usr/bin/env python
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')
django.setup()

from django.contrib.auth.models import User

# Create test user
try:
    user = User.objects.create_user(
        username='test@example.com',
        email='test@example.com',
        password='password123'
    )
    user.first_name = 'Test'
    user.last_name = 'User'
    user.save()
    print('Test user created successfully!')
    print(f'Email: test@example.com')
    print(f'Password: password123')
except Exception as e:
    print(f'Error creating user: {e}')