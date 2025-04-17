from django.db import models
from django.contrib.auth.models import User

class Person(models.Model):
    admin = models.ForeignKey(  # Add this field
        User,
        on_delete=models.CASCADE,
        related_name='persons',
        null=True  # Temporary for migration
    )
    name = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('admin', 'name')  # Prevent duplicate names per user

class FaceImage(models.Model):
    person = models.ForeignKey(
        Person,
        on_delete=models.CASCADE,
        related_name='face_images'
    )
    image = models.ImageField(upload_to='face_images/')
    embedding = models.BinaryField()
    created_at = models.DateTimeField(auto_now_add=True)

