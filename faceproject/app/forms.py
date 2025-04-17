from django import forms
from django.contrib.auth.models import User


class ImageUploadForm(forms.Form):
    person_name = forms.CharField(max_length=255, label="Person's Name")
    image = forms.ImageField(label="Select Image")

class UserForm(forms.ModelForm):
    # Define password fields manually since they're not part of the model fields
    password1 = forms.CharField(label='Password', widget=forms.PasswordInput)
    password2 = forms.CharField(label='Confirm Password', widget=forms.PasswordInput)
    VUID = forms.CharField()

    class Meta:
        model = User
        fields = ['username', 'password1', 'password2','VUID']

    def clean(self):
        cleaned_data = super().clean()
        password1 = cleaned_data.get("password1")
        password2 = cleaned_data.get("password2")
        if password1 and password2 and password1 != password2:
            self.add_error('password2', "Passwords do not match.")
        return cleaned_data

    def save(self, commit=True):
        user = super().save(commit=False)
        # Set the password properly using the set_password method
        user.set_password(self.cleaned_data["password1"])
        if commit:
            user.save()
        return user