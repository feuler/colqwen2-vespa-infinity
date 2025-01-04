import argparse
import os
import subprocess
from pathlib import Path
from vespa.package import (
    ApplicationPackage,
    Field,
    Schema,
    Document,
    HNSW,
    RankProfile,
    Function,
    FieldSet,
    SecondPhaseRanking,
)

def create_vespa_application_package(vespa_app_name):
    # Define the Vespa schema
    colpali_schema = Schema(
        name="pdf_page",
        document=Document(
            fields=[
                Field(name="id", type="string", indexing=["summary", "index"], match=["word"]),
                Field(name="url", type="string", indexing=["summary", "index"]),
                Field(name="title", type="string", indexing=["summary", "index"], match=["text"], index="enable-bm25"),
                Field(name="page_number", type="int", indexing=["summary", "attribute"]),
                Field(name="image", type="raw", indexing=["summary"]),
                Field(name="full_image", type="raw", indexing=["summary"]),
                Field(name="text", type="string", indexing=["summary", "index"], match=["text"], index="enable-bm25"),
                Field(
                    name="embedding",
                    type="tensor<int8>(patch{}, v[16])",
                    indexing=["attribute", "index"],
                    ann=HNSW(
                        distance_metric="hamming",
                        max_links_per_node=32,
                        neighbors_to_explore_at_insert=400,
                    ),
                ),
            ]
        ),
        fieldsets=[
            FieldSet(name="default", fields=["title", "url", "page_number", "text"]),
            FieldSet(name="image", fields=["image"]),
        ],
    )

    # Define rank profiles
    colpali_profile = RankProfile(
        name="default",
        inputs=[("query(qt)", "tensor<float>(querytoken{}, v[128])")],
        functions=[
            Function(
                name="max_sim",
                expression="""
                    sum(
                        reduce(
                            sum(
                                query(qt) * unpack_bits(attribute(embedding)) , v
                            ),
                            max, patch
                        ),
                        querytoken
                    )
                """,
            ),
            Function(name="bm25_score", expression="bm25(title) + bm25(text)"),
        ],
        first_phase="bm25_score",
        second_phase=SecondPhaseRanking(expression="max_sim", rerank_count=10),
    )
    colpali_schema.add_rank_profile(colpali_profile)

    # Add retrieval-and-rerank rank profile
    input_query_tensors = []
    MAX_QUERY_TERMS = 64
    for i in range(MAX_QUERY_TERMS):
        input_query_tensors.append((f"query(rq{i})", "tensor<int8>(v[16])"))

    input_query_tensors.append(("query(qt)", "tensor<float>(querytoken{}, v[128])"))
    input_query_tensors.append(("query(qtb)", "tensor<int8>(querytoken{}, v[16])"))

    colpali_retrieval_profile = RankProfile(
        name="retrieval-and-rerank",
        inputs=input_query_tensors,
        functions=[
            Function(
                name="max_sim",
                expression="""
                    sum(
                        reduce(
                            sum(
                                query(qt) * unpack_bits(attribute(embedding)) , v
                            ),
                            max, patch
                        ),
                        querytoken
                    )
                """,
            ),
            Function(
                name="max_sim_binary",
                expression="""
                    sum(
                      reduce(
                        1/(1 + sum(
                            hamming(query(qtb), attribute(embedding)) ,v)
                        ),
                        max,
                        patch
                      ),
                      querytoken
                    )
                """,
            ),
        ],
        first_phase="max_sim_binary",
        second_phase=SecondPhaseRanking(expression="max_sim", rerank_count=10),
    )
    colpali_schema.add_rank_profile(colpali_retrieval_profile)

    # Create the Vespa application package
    vespa_application_package = ApplicationPackage(
        name=vespa_app_name,
        schema=[colpali_schema],
    )

    return vespa_application_package

def deploy_vespa_app(app_package_path):
    # Deploy the application using Vespa CLI
    deploy_command = f"vespa deploy {app_package_path} --wait 300"

    try:
        subprocess.run(deploy_command, shell=True, check=True)
        print("Vespa application deployed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error deploying Vespa application: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Deploy Vespa application locally")
    parser.add_argument("--vespa_application_name", required=True, help="Vespa application name")
    parser.add_argument("--output_dir", default="application", help="Output directory for the application package")

    args = parser.parse_args()
    vespa_app_name = args.vespa_application_name
    output_dir = args.output_dir

    # Create the Vespa application package
    app_package = create_vespa_application_package(vespa_app_name)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the application package to disk
    app_package.to_files(output_dir)

    # Deploy the application locally
    deploy_vespa_app(os.path.abspath(output_dir))

    print(f"Application deployed locally. You can access it at http://localhost:8080/")

if __name__ == "__main__":
    main()