from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import os
from tqdm.auto import tqdm
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class StampsDataset(Dataset):
    def __init__(self, dataset_dir: str, csv_path: str, processor: CLIPProcessor):
        self.dataset_dir = dataset_dir
        self.df = pd.read_csv(csv_path)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: str):
        img_path = os.path.join(self.dataset_dir, f"{idx}.jpg")
        image = Image.open(img_path).convert("RGB")

        # Process image without resizing
        processed = self.processor(images=image, return_tensors="pt")
        metadata = self.df.iloc[idx].to_dict()

        return processed["pixel_values"][0], idx, metadata


def collate_fn(batch):
    images, ids, metadata = zip(*batch)
    images = torch.stack(images)

    return images, ids, metadata


def load_clip_model(model_name):
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(
        model_name, clean_up_tokenization_spaces=True
    )

    return model, processor, device


def process_batch(images, model, device):
    images = images.to(device)

    with torch.no_grad():
        embeddings = model.get_image_features(pixel_values=images)

    return embeddings.cpu().numpy()


def main():
    # Configuration
    qdrant_url = "https://qdrant.fijit.club:443"
    collection_name = "stamps"
    model_name = "openai/clip-vit-base-patch32"
    csv_path = "./data/stanley_gibbons_colnect.csv"
    dataset_dir = "./data/images"
    batch_size = 32  # Reduced batch size due to varying image sizes
    num_workers = os.cpu_count()  # Adjust based on your CPU
    vector_size = 512

    # Setup
    client = QdrantClient(qdrant_url)
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    model, processor, device = load_clip_model(model_name)

    # Load data

    # Create dataset and dataloader
    dataset = StampsDataset(dataset_dir, csv_path, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Process images and upload to Qdrant
    total_batches = len(dataloader)
    for batch in tqdm(dataloader, total=total_batches, desc="Embedding Stamps"):
        pixel_values, ids, metadata = batch
        print(pixel_values.shape, ids, len(metadata))
        embeddings = process_batch(pixel_values, model, device)

        client.upsert(
            collection_name,
            points=[models.Batch(ids=ids, payloads=metadata, vectors=embeddings)],
        )

        break  # Remove this line to process all images


if __name__ == "__main__":
    main()
