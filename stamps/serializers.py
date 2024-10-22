from rest_framework import serializers

from stamps.models import Stamp


class StampsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Stamp
        fields = "__all__"


class ImageUploadSerializer(serializers.Serializer):
    image = serializers.ImageField(required=True)

    class Meta:
        fields = ["image"]
