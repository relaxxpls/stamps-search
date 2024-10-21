from rest_framework import serializers

from stamps.models import Stamp


class StampsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Stamp
