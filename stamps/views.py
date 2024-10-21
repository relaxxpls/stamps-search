from rest_framework.generics import ListAPIView

from stamps.models import Stamp
from stamps.serializers import StampsSerializer


class StampsListView(ListAPIView):
    serializer_class = StampsSerializer
    queryset = Stamp.objects.all()
