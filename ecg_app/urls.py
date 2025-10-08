from django.urls import path
from . import views

urlpatterns = [
    # Root URL - redirects to login or home based on authentication
    path('', views.index_view, name='index'),
    
    # Authentication URLs
    path('login/', views.login_view, name='login'),
    path('signup/', views.signup_view, name='signup'),
    path('logout/', views.logout_view, name='logout'),
    
    # ECG App URLs (now protected)
    path('home/', views.upload_and_analyze, name='home'),
    path('upload/', views.upload_and_analyze, name='upload_ecg'),
    path('uploaded-files/', views.uploaded_files, name='uploaded_files'),
    path('file/<int:file_id>/', views.file_detail, name='file_detail'),
    path('delete/<int:file_id>/', views.delete_file, name='delete_file'),
    path('ajax/delete/<int:file_id>/', views.delete_file_ajax, name='delete_file_ajax'),
]
