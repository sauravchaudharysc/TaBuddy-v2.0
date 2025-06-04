"""ldap_auth URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from ldap_auth import views
from django.views.decorators.csrf import csrf_exempt
from ldap_auth.views import UserLoginView, UserLogoutView

urlpatterns = [
    path('', UserLoginView.as_view(), name='login'),
    path('success/', views.login_success, name='login_success'),
    path('logout/', UserLogoutView.as_view(), name='logout'),
    path('profile/', views.profile, name='profile'),
    path('login/', views.ldap_login, name='ldap_login'),
    path('login-check/', csrf_exempt(views.ldap_login_check), name='ldap_login_check'),
]

