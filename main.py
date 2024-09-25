from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import pipeline, CLIPProcessor, CLIPModel
from annoy import AnnoyIndex

# Load the CLIP model
model_name = "openai/clip-vit-base-patch32"

model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name, clean_up_tokenization_spaces=True)

# Load the Annoy index
annoy_index = AnnoyIndex(512, "angular")
annoy_index.load("embeddings/uk_stamps.ann")


@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> "DetectionResult":
        return cls(
            score=detection_dict["score"],
            label=detection_dict["label"],
            box=BoundingBox(
                xmin=detection_dict["box"]["xmin"],
                ymin=detection_dict["box"]["ymin"],
                xmax=detection_dict["box"]["xmax"],
                ymax=detection_dict["box"]["ymax"],
            ),
        )


def detect(
    image: Image.Image,
    labels: List[str],
    threshold: float = 0.3,
    detector_id: Optional[str] = "IDEA-Research/grounding-dino-tiny",
) -> List[Dict[str, Any]]:
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    object_detector = pipeline(
        model=detector_id, task="zero-shot-object-detection", device=device
    )

    labels = [label if label.endswith(".") else label + "." for label in labels]

    results = object_detector(image, candidate_labels=labels, threshold=threshold)
    results = [DetectionResult.from_dict(result) for result in results]

    return results


def detect_stamps(image: Image.Image) -> List[Tuple[Image.Image, DetectionResult]]:
    """
    Return a list of cropped images and their corresponding detection results.
    """

    detections = detect(
        image,
        ["a postage stamp."],
        threshold=0.1,
        detector_id="IDEA-Research/grounding-dino-base",
    )

    cropped_images = []
    for detection in detections:
        box = detection.box
        cropped_image = image.crop(box.xyxy)
        cropped_images.append((cropped_image, detection))

    return cropped_images


def extract_stamp_embedding(image: Image.Image) -> np.array:

    return


def find_similar_stamps(
    image: Image.Image, num_matches: int = 5
) -> List[Dict[str, Any]]:
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)

    embedding = embeddings.squeeze().numpy()

    indices, distances = annoy_index.get_nns_by_vector(
        embedding, num_matches, include_distances=True
    )

    similar_images = [
        {"idx": idx, "distance": distances[i]} for i, idx in enumerate(indices)
    ]

    return similar_images


def identify_stamps(image_path: str) -> List[DetectionResult]:
    """
    Identify postage stamps in an image.
    """
    image = Image.open(image_path).convert("RGB")
    detected_stamps = detect_stamps(image)

    for cropped_image, _ in detected_stamps:
        similar_images = find_similar_stamps(cropped_image)
        # plot in a grid pattern with rows as image followed by its similar images with captions as distance

        # "path": f"{dataset_dir}/{idx}.jpg",
