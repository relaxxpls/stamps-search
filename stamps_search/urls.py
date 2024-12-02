from django.contrib import admin
from django.urls import include, path

admin.site.site_header = "Stamps Search Admin"
admin.site.site_title = "Stamps Search Admin"

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("stamps.urls")),
]

urlpatterns = [path("stamps/", include(urlpatterns))]
