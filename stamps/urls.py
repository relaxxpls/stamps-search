from django.urls import path

from stamps.views import StampsListView


urlpatterns = [
    path("", StampsListView.as_view(), name="stamp-list"),
]
