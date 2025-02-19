"""A Union app that uses sklearn and Streamlit"""

import os
from union import Artifact, ImageSpec, Resources
from union.app import App, Input

# Define the artifact that holds the iris model.
KnnModelArtifact = Artifact(name="knn_model")

# Define the container image including the required packages.
image_spec = ImageSpec(
    name="union-serve-iris-streamlit",
    packages=[
        "scikit-learn==1.6.0",
        "union-runtime>=0.1.10",
        "streamlit",  # For the UI
    ],
    registry=os.getenv("REGISTRY"),
)

# Create the Union Serving App.
streamlit_app = App(
    name="simple-streamlit-iris",
    inputs=[
        Input(
            name="sklearn_model",
            value=KnnModelArtifact.query(),
            download=True,  # The model artifact is downloaded when the container starts.
        )
    ],
    container_image=image_spec,
    limits=Resources(cpu="1", mem="1Gi"),
    port=8082,
    include=["./main.py"],  # Include your Streamlit code.
    args=["streamlit", "run", "main.py", "--server.port", "8082"],
    # requires_auth=False # Uncomment to make app public.
)

# union deploy apps app.py simple-streamlit-iris
