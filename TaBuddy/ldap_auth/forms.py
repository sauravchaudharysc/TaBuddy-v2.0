from django import forms


class UserLoginForm(forms.Form):
    username = forms.CharField(max_length=64,
    min_length=1,widget=forms.TextInput(
        attrs={
            'class':'form-control',
            'placeholder':'cse username',
        }
    ))

    password = forms.CharField(
        widget = forms.PasswordInput(
            attrs={
                'class':'form-control',
                'placeholder':'********',
            }
        )
    )

class TaskSubmissionForm(forms.Form):
    problem_statement = forms.CharField(
        widget=forms.Textarea(attrs={'placeholder': 'Problem Statement…','rows': 5}))
    rubric = forms.CharField(
        widget=forms.Textarea(attrs={'placeholder': 'Rubrics…','rows': 5}))
    student_codes = forms.FileField(
        widget=forms.ClearableFileInput(attrs={'multiple': True}),
        help_text="Upload one or more student submission files"
    )