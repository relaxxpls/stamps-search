from typing import List

import torch
from PIL import Image
from transformers import pipeline, CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from tqdm import tqdm

from models.models import DetectionResult, BoundingBox


class StampSearch:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient("https://qdrant.fijit.club:443")

        # Detection model
        self.detector_id = "IDEA-Research/grounding-dino-base"

        # Embedding model
        # Base
        self.embedding_collection_name = "stamps-clip-base"
        self.embedding_model_name = "openai/clip-vit-base-patch32"
        # Large
        # self.embedding_collection_name = "stamps-clip-large"
        # self.embedding_model_name = "openai/clip-vit-large-patch14"

    def search(self, image: Image.Image):
        """
        Identify postage stamps in an image and visualize the results.
        """
        all_detections = self.__detect_stamps(image, threshold=0.1)
        detections = self.__remove_overlapping_detections(
            all_detections, overlap_threshold=0.9
        )

        queries = []
        for detection in tqdm(detections):
            box = detection.box
            cropped_image = image.crop(box.xyxy)
            query_res = self.__find_similar_stamps(cropped_image)
            queries.append(query_res)

        return detections, queries

    def __detect_stamps(
        self,
        image: Image.Image,
        labels: List[str] = ["a postage stamp."],
        threshold=0.3,
    ) -> List[DetectionResult]:
        """
        Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
        Return a list of detection results.
        """

        labels = [label if label.endswith(".") else label + "." for label in labels]
        object_detector = pipeline(
            model=self.detector_id,
            task="zero-shot-object-detection",
            device=self.device,
        )
        results = object_detector(image, candidate_labels=labels, threshold=threshold)
        results = [DetectionResult.from_dict(result) for result in results]

        return results

    def __remove_overlapping_detections(
        self, detections: List[DetectionResult], overlap_threshold=0.9
    ) -> List[DetectionResult]:
        """
        Remove overlapping detections, keeping the smaller box when one box mostly contains another.
        """
        sorted_detections = sorted(detections, key=lambda x: x.box.area)
        kept_detections = []

        for detection in sorted_detections:
            is_duplicate = False

            for kept_detection in kept_detections:
                overlap = BoundingBox.calc_overlap(detection.box, kept_detection.box)
                if overlap > overlap_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept_detections.append(detection)

        return kept_detections

    def __find_similar_stamps(self, image: Image.Image, num_matches=5):
        # Load the CLIP model
        model = CLIPModel.from_pretrained(self.embedding_model_name)
        model.to(self.device)

        processor = CLIPProcessor.from_pretrained(self.embedding_model_name)
        inputs = processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            embeddings = model.get_image_features(**inputs)

        embedding = embeddings.cpu().squeeze().numpy()

        # ? Use Qdrant to find similar stamps
        results = self.qdrant_client.query_points(
            self.embedding_collection_name, query=embedding, limit=num_matches
        )

        return results
