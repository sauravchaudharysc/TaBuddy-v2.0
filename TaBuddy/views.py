from django.shortcuts import render

def home(request):
    return render(request, 'home.html')  # Or 'your_app/home.html' if needed
