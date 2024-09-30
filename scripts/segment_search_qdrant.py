from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Tuple
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import pipeline, CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from tqdm.auto import tqdm

# Load the CLIP model
model_name = "openai/clip-vit-base-patch32"

model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name, clean_up_tokenization_spaces=True)

# Initialize Qdrant client
qdrant_client = QdrantClient("https://qdrant.fijit.club:443")
collection_name = "stamps"


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


def find_similar_stamps(
    image: Image.Image, num_matches: int = 5
) -> List[Dict[str, Any]]:
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)

    embedding = embeddings.squeeze().numpy()

    # Use Qdrant to find similar stamps
    results = qdrant_client.query_points(collection_name, query=embedding)

    similar_images = [{"idx": hit.id, "distance": hit.score} for hit in results.points]

    return similar_images


def resize_image(image: Image.Image, max_size: int = 200) -> Image.Image:
    """Resize image while maintaining aspect ratio"""
    ratio = max_size / max(image.size)
    new_size = tuple([int(x * ratio) for x in image.size])
    return image.resize(new_size, Image.Resampling.LANCZOS)


def identify_stamps(image_path: str, dataset_dir: str) -> None:
    """
    Identify postage stamps in an image and visualize the results.
    """
    image = Image.open(image_path).convert("RGB")
    detected_stamps = detect_stamps(image)

    for i, (cropped_image, _) in tqdm(
        enumerate(detected_stamps), desc="Detecting stamps"
    ):
        similar_images = find_similar_stamps(cropped_image)

        # Create a new figure for each detected stamp
        fig, axes = plt.subplots(2, 3, figsize=(10, 6))
        fig.suptitle(f"Detected Stamp {i + 1}")

        # Plot the detected stamp
        axes[0, 0].imshow(resize_image(cropped_image))
        axes[0, 0].set_title("Detected Stamp")
        axes[0, 0].axis("off")

        # Plot similar stamps
        for j, similar in enumerate(similar_images):
            similar_image_path = os.path.join(dataset_dir, f"{similar['idx']}.jpg")
            similar_image = Image.open(similar_image_path).convert("RGB")

            row = (j + 1) // 3
            col = (j + 1) % 3

            axes[row, col].imshow(resize_image(similar_image))
            axes[row, col].set_title(f"Distance: {similar['distance']:.4f}")
            axes[row, col].axis("off")

        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    image_path = "./data/album/album1.png"
    dataset_dir = "./data/images"
    identify_stamps(image_path, dataset_dir)
