from flytekit import ImageSpec

image = ImageSpec(
    packages=[
        "union==0.1.109",
        "flytekit==1.13.13",
        "scikit-learn==1.4.1.post1",
        "matplotlib==3.8.3",
        "seaborn==0.13.2",
        "joblib==1.3.2",
        "huggingface_hub==0.24.0",
        "pyarrow==16.0.0",
        "python-dotenv"
    ],
)

