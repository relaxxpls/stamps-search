from dataclasses import asdict
from rest_framework.generics import ListAPIView
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from PIL import Image
from django.forms.models import model_to_dict

from models.core import StampSearch
from stamps.models import Stamp
from stamps.serializers import StampsSerializer, ImageUploadSerializer


class StampsListView(ListAPIView):
    serializer_class = StampsSerializer
    queryset = Stamp.objects.all()


class IdentifyImage(APIView):
    def post(self, request):
        serializer = ImageUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            image = Image.open(serializer.validated_data["image"]).convert("RGB")
            stamp_search = StampSearch()
            detections, queries = stamp_search.search(image)

            colnect_ids = [
                int(point.payload["colnect_id"])
                for query in queries
                for point in query.points
            ]

            stamps = Stamp.objects.filter(id__in=colnect_ids)
            stamp_dict = {stamp.id: stamp for stamp in stamps}

            res = []
            for detection, query in zip(detections, queries):
                similar_stamps = []
                for point in query.points:
                    stamp = model_to_dict(stamp_dict[int(point.payload["colnect_id"])])
                    stamp["score"] = point.score
                    similar_stamps.append(stamp)

                res.append(
                    {"detection": asdict(detection), "similar_stamps": similar_stamps}
                )

            return Response(res, status=status.HTTP_200_OK)

        except Exception as e:
            print(e)
            return Response(
                "Internal Error Occured", status=status.HTTP_400_BAD_REQUEST
            )
