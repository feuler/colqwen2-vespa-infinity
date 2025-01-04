from openwebui import Pipeline, Component
from vespa.application import Vespa
from vespa.io import VespaQueryResponse
import numpy as np
import base64
from PIL import Image
import io
from typing import List, Dict, Any
import asyncio
from dataclasses import dataclass
import json

@dataclass
class VespaConfig:
    vespa_url: str = "http://localhost:8080"
    max_hits: int = 3
    max_query_terms: int = 64

class VespaImageRetriever(Component):
    def __init__(self, config: VespaConfig):
        super().__init__()
        self.config = config
        self.app = Vespa(url=config.vespa_url)
        # Verify connection
        status = self.app.get_application_status()
        if status.status_code != 200:
            raise ConnectionError(f"Failed to connect to Vespa at {config.vespa_url}")

    async def query_vespa(self, query: str, embedding: np.ndarray) -> List[Dict[str, Any]]:
        float_query_embedding = {k: v.tolist() for k, v in enumerate(embedding)}
        binary_query_embeddings = {}

        # Convert to binary embeddings
        for k, v in float_query_embedding.items():
            binary_vector = (np.packbits(np.where(np.array(v) > 0, 1, 0))
                           .astype(np.int8)
                           .tolist())
            binary_query_embeddings[k] = binary_vector
            if len(binary_query_embeddings) >= self.config.max_query_terms:
                break

        # Prepare query tensors
        query_tensors = {
            "input.query(qtb)": binary_query_embeddings,
            "input.query(qt)": float_query_embedding,
        }

        # Add nearest neighbor query tensors
        for i in range(len(binary_query_embeddings)):
            query_tensors[f"input.query(rq{i})"] = binary_query_embeddings[i]

        # Build nearest neighbor query
        nn_queries = [
            f"({{targetHits:20}}nearestNeighbor(embedding,rq{i}))"
            for i in range(len(binary_query_embeddings))
        ]
        nn_query = " OR ".join(nn_queries)

        async with self.app.asyncio() as session:
            response: VespaQueryResponse = await session.query(
                body={
                    **query_tensors,
                    "presentation.timing": True,
                    "yql": f"select documentid, title, url, image, page_number from pdf_page where {nn_query}",
                    "ranking.profile": "retrieval-and-rerank",
                    "timeout": 120,
                    "hits": self.config.max_hits,
                }
            )

            if not response.is_successful():
                raise Exception(f"Vespa query failed: {response.json}")

            # Extract images and metadata
            results = []
            for hit in response.hits:
                image_data = base64.b64decode(hit["fields"]["image"])
                image = Image.open(io.BytesIO(image_data))

                results.append({
                    "image": image,
                    "title": hit["fields"]["title"],
                    "page": hit["fields"]["page_number"],
                    "url": hit["fields"]["url"],
                    "score": hit["relevance"]
                })

            return results

    async def process(self, query: str, embedding: np.ndarray) -> List[Dict[str, Any]]:
        return await self.query_vespa(query, embedding)

class Qwen2VLAnalyzer(Component):
    def __init__(self, model_name: str = "qwen2vl"):
        super().__init__()
        self.model_name = model_name

    async def process(self, query: str, vespa_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for item in vespa_results:
            # Convert PIL Image to format expected by Qwen-VL
            image = item["image"]

            # Construct prompt for the vision model
            prompt = f"Query: {query}\nAnalyze this image from document: {item['title']}, page {item['page']}"

            # Call Qwen-VL model (implementation depends on your setup)
            analysis = await self.analyze_with_qwen(image, prompt)

            results.append({
                **item,
                "analysis": analysis
            })

        return results

    async def analyze_with_qwen(self, image: Image.Image, prompt: str) -> str:
        # Implement your Qwen-VL model call here
        # This is a placeholder - replace with actual implementation
        return "Qwen-VL analysis placeholder"

class VespaQwenPipeline(Pipeline):
    def __init__(self, vespa_config: VespaConfig):
        super().__init__()
        self.vespa_retriever = VespaImageRetriever(vespa_config)
        self.qwen_analyzer = Qwen2VLAnalyzer()

    async def process(self, query: str, embedding: np.ndarray) -> List[Dict[str, Any]]:
        # Retrieve images from Vespa
        vespa_results = await self.vespa_retriever.process(query, embedding)

        # Analyze images with Qwen-VL
        analyzed_results = await self.qwen_analyzer.process(query, vespa_results)

        return analyzed_results

# Example usage
async def main():
    # Configure pipeline
    config = VespaConfig(
        vespa_url="http://localhost:8080",
        max_hits=3
    )

    pipeline = VespaQwenPipeline(config)

    # Example query and embedding (replace with actual embedding)
    query = "HPC SDK 23.11"
    embedding = np.random.randn(512)  # Replace with actual embedding dimension

    # Process query through pipeline
    results = await pipeline.process(query, embedding)

    # Print results
    for idx, result in enumerate(results, 1):
        print(f"\nResult {idx}:")
        print(f"Title: {result['title']}")
        print(f"Page: {result['page']}")
        print(f"Score: {result['score']:.2f}")
        print(f"Analysis: {result['analysis']}")

if __name__ == "__main__":
    asyncio.run(main())
