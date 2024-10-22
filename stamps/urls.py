from django.urls import path

from stamps.views import StampsListView, IdentifyImage


urlpatterns = [
    path("", StampsListView.as_view(), name="stamp-list"),
    path("search", IdentifyImage.as_view(), name="stamp-search"),
]
