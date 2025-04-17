from django.urls import path
from .views import *


urlpatterns = [
    path('',home, name='home'),
    path('upload/', upload_image, name='upload'),
    path('recognize/', recognize_page, name='recognize'),
    path('video-feed/', face_recognition_view, name='video_feed'),
    path('stop-recognition/', stop_recognition, name='stop_recognition'),
    path('gallery/',gallery_view, name='gallery'),
    path('register/', register_view, name='register'),
    path('login/', login_view, name='login'),
    path('logout/', logoutview, name='logout'),
    path('delete-image/<int:image_id>/', Delete_picture_view, name='delete-image'),
]