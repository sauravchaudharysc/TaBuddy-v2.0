import ldap
from django_auth_ldap.config import LDAPSearch, PosixGroupType
import os


AUTH_USER_MODEL = 'ldap_auth.CustomUser'
AUTH_LDAP_CREATE_USERS = True

# Baseline configuration.
AUTH_LDAP_SERVER_URI = os.getenv('AUTH_LDAP_SERVER_URI', "ldap://1.ldap.cse.iitb.ac.in:389")

# Anonymous bind is allowed 
AUTH_LDAP_BIND_DN = ''
AUTH_LDAP_BIND_PASSWORD = ''

# Search for users under the folder /People in the cse.iitb.ac.in domain.
AUTH_LDAP_USER_SEARCH = LDAPSearch(
    'ou=People,dc=cse,dc=iitb,dc=ac,dc=in',
    ldap.SCOPE_SUBTREE,
    '(uid=%(user)s)'
)

AUTH_LDAP_GROUP_SEARCH = LDAPSearch(
    'ou=Groups,dc=cse,dc=iitb,dc=ac,dc=in',
    ldap.SCOPE_SUBTREE,
    '(objectClass=*)',
)

# The LDAP attributes are mapped to Django user model fields.
AUTH_LDAP_USER_ATTR_MAP = {
    'username': 'uid',
    'first_name': 'cn',
    'last_name': 'sn',
    'email': 'mail',
}


# For group-based access control .
# Cse IITB uses PosixGroupType for group membership.
AUTH_LDAP_GROUP_TYPE = PosixGroupType(name_attr='cn') 

AUTH_LDAP_MIRROR_GROUPS_EXCEPT = ['moderator','webfac','external']
# To Check LDAP to see if the user is a member of a group.
AUTH_LDAP_FIND_GROUP_PERMS = True
# Cache the group memberships
AUTH_LDAP_CACHE_GROUPS = True


# Checks User Membership in LDAP groups, assigns given mentioned permissions to the user.
AUTH_LDAP_USER_FLAGS_BY_GROUP = {
    "is_staff" : ["cn=staff,ou=Groups,dc=cse,dc=iitb,dc=ac,dc=in","cn=webteam,ou=Groups,dc=cse,dc=iitb,dc=ac,dc=in"],
    "is_superuser" : "cn=webteam,ou=Groups,dc=cse,dc=iitb,dc=ac,dc=in"
}

# Every time a user logs in, the LDAP server is checked for the latest information.
AUTH_LDAP_ALWAYS_UPDATE_USER = True

AUTHENTICATION_BACKENDS = [
    'django_auth_ldap.backend.LDAPBackend',
    'django.contrib.auth.backends.ModelBackend',
]