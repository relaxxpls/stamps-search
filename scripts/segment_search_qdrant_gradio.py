from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Tuple
import os

import gradio as gr
import torch
import numpy as np
from PIL import Image
from transformers import pipeline, CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize Qdrant client
qdrant_client = QdrantClient("https://qdrant.fijit.club:443")

# # Large
# embedding_collection_name = "stamps2"
# embedding_model_name = "openai/clip-vit-large-patch14"

# Base
embedding_collection_name = "stamps"
embedding_model_name = "openai/clip-vit-base-patch32"


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
    # Load the CLIP model
    model = CLIPModel.from_pretrained(embedding_model_name)
    model.to(device)
    processor = CLIPProcessor.from_pretrained(
        embedding_model_name, clean_up_tokenization_spaces=True
    )

    inputs = processor(images=image, return_tensors="pt")
    inputs.to(device)

    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)

    embedding = embeddings.cpu().squeeze().numpy()

    # Use Qdrant to find similar stamps
    results = qdrant_client.query_points(
        embedding_collection_name, query=embedding, limit=num_matches
    )

    similar_images = [{"idx": hit.id, "distance": hit.score} for hit in results.points]

    return similar_images


def identify_stamps(image_np: np.ndarray) -> List[List[Image.Image]]:
    """
    Identify postage stamps in an image and visualize the results.
    """

    dataset_dir = "./data/images"  # Update this path to your dataset directory
    image = Image.fromarray(image_np)
    detected_stamps = detect_stamps(image)

    results = []
    for i, (cropped_image, _) in enumerate(detected_stamps):
        similar_images = find_similar_stamps(cropped_image)

        # Add the detected stamp
        results.append((np.array(cropped_image), f"[{i + 1}] Detected Stamp"))

        # Add similar stamps
        for j, similar in enumerate(similar_images):
            similar_image_path = os.path.join(dataset_dir, f"{similar['idx']}.jpg")
            similar_image = Image.open(similar_image_path).convert("RGB")

            similar_caption = (
                f"[{i + 1}] Similar {j + 1} (Distance: {similar['distance']:.4f})"
            )
            results.append((np.array(similar_image), similar_caption))

    return results


def create_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Stamp Identifier")
        with gr.Row():
            input_image = gr.Image()
            output_gallery = gr.Gallery(
                label="Detected Stamps and Similar Images", object_fit="contain"
            )

        submit_btn = gr.Button("Identify Stamps")

        submit_btn.click(
            fn=identify_stamps, inputs=[input_image], outputs=[output_gallery]
        )

    return demo


if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(share=True)
