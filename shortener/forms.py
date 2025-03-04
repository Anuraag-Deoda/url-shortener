from django import forms
from .models import URL
import validators


class URLShortenerForm(forms.ModelForm):
    original_url = forms.URLField(
        widget=forms.URLInput(attrs={
            'class': 'form-control form-control-lg',
            'placeholder': 'Enter your long URL here...',
            'aria-label': 'URL to shorten',
        }),
        label='',
    )
    
    class Meta:
        model = URL
        fields = ['original_url']
        
    def clean_original_url(self):
        url = self.cleaned_data['original_url']
        if not validators.url(url):
            raise forms.ValidationError('Please enter a valid URL including http:// or https://')
        return url
