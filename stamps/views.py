from rest_framework.generics import ListAPIView
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from PIL import Image

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

            stamps = Stamp.objects.filter(colnect_id__in=colnect_ids)
            stamp_dict = {stamp.colnect_id: stamp for stamp in stamps}

            res = []
            for detection, query in zip(detections, queries):
                similar_stamps = []
                for point in query.points:
                    stamp = stamp_dict[int(point.payload["colnect_id"])].__dict__
                    stamp["score"] = point.score
                    similar_stamps.append(stamp)
                res.append({"detection": detection, "similar_stamps": similar_stamps})

            return Response(res, status=status.HTTP_200_OK)

        except Exception as e:
            print(e)
            return Response(
                "Internal Error Occured", status=status.HTTP_400_BAD_REQUEST
            )
