from django.contrib.auth import authenticate, login, logout
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.views.generic.edit import FormView, View
import json
from .forms import UserLoginForm
from django.urls import reverse_lazy, reverse
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required


import logging
logger = logging.getLogger(__name__)

class UserLoginView(FormView):
    form_class = UserLoginForm
    template_name = 'ldap_auth/login.1.html'
    success_url = reverse_lazy('login_success')

    def get(self, request, *args, **kwargs):
        next_ = request.GET.get('next','')
        if next_ == '':
            next_ = self.success_url
        if request.user.is_authenticated:
            return redirect(next_)

        return render(request, self.template_name, {'form': self.form_class, 'isLoggedIn': request.user.is_authenticated})

    def post(self, request, *args, **kwargs):
        form = self.form_class(request.POST)
        next_ = request.POST.get('next', self.success_url)

        if next_ == '':
            next_ = self.success_url
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']

        user = authenticate(username=username, password=password)
        if user is not None:
            # print(hasattr(user, 'ldap_user'))
            login(request, user)
            logger.info('User {} {} with username {} is sucessfully logged into system'.format(request.user.first_name,
                                                                                               request.user.last_name, request.user.username))
            return redirect(next_)
        else:
            # print(form.errors.as_json)
            logger.error(
                'Unknown user / Authentication failed for {}'.format(username))
            messages.error(
                request, 'Unknown user / Authentication failed for {}'.format(username))
            return redirect(reverse('logout'))

    def __init__(self):
        print('Login View Loaded')
        return


@login_required(login_url=reverse_lazy('login'))
def login_success(request):
    return render(request, 'ldap_auth/success.html')

@login_required(login_url=reverse_lazy('login'))
def profile(request):
    return render(request, 'ldap_auth/profile.html')


class UserLogoutView(View):
    def post(self, request, *args, **kwargs):
        if request.user:
            logger.info('User {} {} with username {} is sucessfully logged out of system'.format(self.request.user.first_name,
                                                                                                 self.request.user.last_name, self.request.user.username))
            logout(request)
        return redirect('logout')

    def get(self, request, *args, **kwargs):
        if request.user:
            logout(request)
        return redirect('login')

    def __init__(self):
        print('logout')


# The below are simple example functions for LDAP login handling 
@csrf_exempt
@require_POST
def ldap_login(request):
    try:
        data = json.loads(request.body)
        username = data.get('username')
        password = data.get('password')
        if not username or not password:
            return JsonResponse({'error': 'Username and password required.'}, status=400)

        user = authenticate(username=username, password=password)
        if user is not None:
            login(request, user)
            return JsonResponse({'success': True, 'message': 'Logged in successfully.'})
        else:
            return JsonResponse({'success': False, 'error': 'Invalid credentials.'}, status=401)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def ldap_login_check(request):
    if request.method == 'GET':
        return JsonResponse({'success': True, 'message': 'LDAP login endpoint is reachable.'})
    else:
        return JsonResponse({'error': 'Method not allowed.'}, status=405)
    
