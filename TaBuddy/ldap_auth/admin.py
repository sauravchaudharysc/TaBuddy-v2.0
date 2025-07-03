from django.contrib import admin, messages
from django.contrib.auth.forms import AdminPasswordChangeForm, UserChangeForm
from django.views.decorators.debug import sensitive_post_parameters
from django.utils.decorators import method_decorator
from django.template.response import TemplateResponse
from django.contrib.admin.utils import unquote
from django.core.exceptions import PermissionDenied
from django.http import Http404, HttpResponseRedirect
from django.utils.html import escape
from django.utils.translation import gettext, gettext_lazy as _
from django.contrib.admin.options import IS_POPUP_VAR
from django.urls import path, reverse
from django.contrib.auth import update_session_auth_hash
from ldap_auth.models import User

sensitive_post_parameters_m = method_decorator(sensitive_post_parameters())

@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    model = User
    list_display = ('email', 'first_name', 'last_name', 'is_staff', 'is_active')
    list_filter = ('is_staff', 'is_active', 'is_superuser', 'groups', 'is_ai_admin')
    search_fields = ('email', 'first_name', 'last_name')
    ordering = ('email',)
    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        ('Personal Info', {'fields': ('first_name', 'last_name')}),
        ('Permissions', {'fields': ('is_ai_admin', 'is_staff', 'is_active', 'is_superuser', 'groups', 'user_permissions')}),
        ('Important dates', {'fields': ('last_login', 'date_joined')}),
    )
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'password1', 'password2', 'is_staff', 'is_active', 'is_ai_admin')}
        ),
    )

    change_user_password_template = None
    form = UserChangeForm
    change_password_form = AdminPasswordChangeForm

    def get_urls(self):
        return [
            path(
                '<id>/password/',
                self.admin_site.admin_view(self.user_change_password),
                name='auth_user_password_change',
            ),
        ] + super().get_urls()

    @sensitive_post_parameters_m
    def user_change_password(self, request, id, form_url=''):
        user = self.get_object(request, unquote(id))
        if not self.has_change_permission(request, user):
            raise PermissionDenied
        if user is None:
            raise Http404(
                _('%(name)s object with primary key %(key)r does not exist.') % {
                    'name': self.model._meta.verbose_name,
                    'key': escape(id),
                }
            )

        # 1. Instantiate form for both GET and POST
        if request.method == 'POST':
            form = self.change_password_form(user, request.POST)
        else:
            form = self.change_password_form(user)

        # 2. On valid POST, save + redirect
        if request.method == 'POST' and form.is_valid():
            form.save()
            change_message = self.construct_change_message(request, form, None)
            self.log_change(request, user, change_message)
            messages.success(request, gettext('Password changed successfully.'))
            update_session_auth_hash(request, form.user)
            return HttpResponseRedirect(
                reverse(
                    '%s:%s_%s_change' % (
                        self.admin_site.name,
                        user._meta.app_label,
                        user._meta.model_name,
                    ),
                    args=(user.pk,),
                )
            )

        # 3. For GET **or** POST with errors: render form with errors
        fieldsets = [(None, {'fields': list(form.base_fields)})]
        adminForm = admin.helpers.AdminForm(form, fieldsets, {})
        context = {
            **self.admin_site.each_context(request),
            'title': _('Change password: %s') % escape(user.get_username()),
            'adminForm': adminForm,
            'form': form,
            'form_url': form_url,
            'is_popup': IS_POPUP_VAR in request.POST or IS_POPUP_VAR in request.GET,
            'add': True,
            'change': False,
            'has_delete_permission': False,
            'has_change_permission': True,
            'opts': self.model._meta,
            'original': user,
            'save_as': False,
            'show_save': True,
        }
        request.current_app = self.admin_site.name
        return TemplateResponse(
            request,
            self.change_user_password_template or
            'admin/auth/user/change_password.html',
            context,
        )
