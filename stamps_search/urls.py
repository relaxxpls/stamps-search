from django.contrib import admin
from django.urls import include, path

admin.site.site_header = "ResoBin Admin"
admin.site.site_title = "ResoBin Admin"

urlpatterns = [
    path("admin/", admin.site.urls),
    path("stamps/", include("stamps.urls")),
]
