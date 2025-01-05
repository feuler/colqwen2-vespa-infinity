#!/usr/bin/env python3

import argparse
from typing import List
import os
import json
import hashlib
import re
import numpy as np
from tqdm import tqdm
from pdf2image import convert_from_path
from pypdf import PdfReader
from vespa.application import Vespa
from vespa.io import VespaResponse
from dotenv import load_dotenv
from openai import OpenAI
import base64
from torch.utils.data import DataLoader
from PIL import Image
import io
import torch
import contextlib
import sys

load_dotenv()

# Configure OpenAI client
client = OpenAI(
    base_url="http://localhost:7997",
    api_key="sk-1234"
)

def sanitize_text(text):
    # Remove control characters and non-printable characters
    return re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)

def get_image_embedding(image: Image.Image) -> np.ndarray:
    # Convert PIL Image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    response = client.embeddings.create(
        model="colqwen2",
        input=["data:image/jpeg;base64," + base64_image],
        extra_body={"modality": "image"},
        encoding_format="float"
    )

    # Extract and process embedding
    embedding = response.data[0].embedding
    embedding_array = np.array(embedding, dtype=np.float32)
    return embedding_array

def process_images_batch(images: List[Image.Image]) -> List[np.ndarray]:
    embeddings = []
    for image in images:
        embedding = get_image_embedding(image)
        embeddings.append(embedding)
    return embeddings

def main():
    parser = argparse.ArgumentParser(description="Feed data into local Vespa application")
    parser.add_argument(
        "--application_name",
        required=True,
        default="colpalidemo",
        help="Vespa application name",
    )
    parser.add_argument(
        "--vespa_schema_name",
        required=True,
        default="pdf_page",
        help="Vespa schema name",
    )
    parser.add_argument(
        "--pdf_folder",
        required=True,
        help="Path to folder containing PDF files",
    )
    args = parser.parse_args()

    vespa_app_url = "http://localhost:8080"  # Assuming local Vespa deployment
    application_name = args.application_name
    schema_name = args.vespa_schema_name
    pdf_folder = args.pdf_folder

    # Instantiate Vespa connection
    app = Vespa(url=vespa_app_url)
    app.get_application_status()

    def get_pdf_images(pdf_path):
        reader = PdfReader(pdf_path)
        page_texts = []
        for page in reader.pages:
            text = page.extract_text()
            sanitized_text = sanitize_text(text)
            page_texts.append(sanitized_text)
        
        # Suppress warnings from pdf2image
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stderr(devnull):
                images = convert_from_path(pdf_path)
        
        assert len(images) == len(page_texts)
        return (images, page_texts)

    # Get all PDF files from the specified folder
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]

    # Check if vespa_feed.json exists
    if os.path.exists("vespa_feed.json"):
        print("Loading vespa_feed from vespa_feed.json")
        with open("vespa_feed.json", "r") as f:
            vespa_feed_saved = json.load(f)
        vespa_feed = []
        for doc in vespa_feed_saved:
            put_id = doc["put"]
            fields = doc["fields"]
            parts = put_id.split("::")
            document_id = parts[1] if len(parts) > 1 else ""
            page = {"id": document_id, "fields": fields}
            vespa_feed.append(page)
    else:
        print("Generating vespa_feed")
        vespa_feed = []
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_folder, pdf_file)
            title = os.path.basename(pdf_file)
            url = pdf_path

            page_images, page_texts = get_pdf_images(pdf_path)

            # Convert PIL images to numpy arrays
            page_images_np = [np.array(image) for image in page_images]

            # Process images in batches using the API
            dataloader = DataLoader(
                page_images_np,
                batch_size=2,
                shuffle=False,
            )

            page_embeddings = []
            for batch_images in tqdm(dataloader):
                # Convert tensors back to numpy arrays
                batch_images_np = [image.numpy() for image in batch_images]
                batch_embeddings = process_images_batch([Image.fromarray(image) for image in batch_images_np])
                page_embeddings.extend(batch_embeddings)

            for page_number, (page_text, embedding, image) in enumerate(
                zip(page_texts, page_embeddings, page_images)
            ):
                # Convert image to base64
                buffered = io.BytesIO()
                scaled_image = image.copy()
                scaled_image.thumbnail((640, 640))
                scaled_image.save(buffered, format="JPEG")
                base_64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

                # Full image
                buffered_full = io.BytesIO()
                image.save(buffered_full, format="JPEG")
                base_64_full_image = base64.b64encode(buffered_full.getvalue()).decode('utf-8')

                # Convert embedding to binary format for Vespa
                embedding_dict = dict()
                for idx, patch_embedding in enumerate(embedding):
                    binary_vector = (
                        np.packbits(np.where(patch_embedding > 0, 1, 0))
                        .astype(np.int8)
                        .tobytes()
                        .hex()
                    )
                    embedding_dict[idx] = binary_vector

                id_hash = hashlib.md5(f"{url}_{page_number}".encode()).hexdigest()
                page = {
                    "id": id_hash,
                    "fields": {
                        "id": id_hash,
                        "url": url,
                        "title": title,
                        "page_number": page_number + 1,
                        "image": base_64_image,
                        "full_image": base_64_full_image,
                        "text": page_text,
                        "embedding": embedding_dict,
                    },
                }
                vespa_feed.append(page)

        # Save vespa_feed to vespa_feed.json
        vespa_feed_to_save = []
        for page in vespa_feed:
            document_id = page["id"]
            put_id = f"id:{application_name}:{schema_name}::{document_id}"
            vespa_feed_to_save.append({"put": put_id, "fields": page["fields"]})
        with open("vespa_feed.json", "w") as f:
            json.dump(vespa_feed_to_save, f)

    def callback(response: VespaResponse, id: str):
        if not response.is_successful():
            print(
                f"Failed to feed document {id} with status code {response.status_code}: Reason {response.get_json()}"
            )
        else:
            print(f"feeding completed successfully")

    # Feed data into Vespa
    app.feed_iterable(vespa_feed, schema=schema_name, callback=callback)

if __name__ == "__main__":
    main()
