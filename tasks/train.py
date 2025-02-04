import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from flytekit.core.artifact import Artifact
from typing_extensions import Annotated
from flytekit import Deck, Resources, task, FlyteFile, current_context
from containers import image
import joblib
from pathlib import Path

# Define Artifacts
TrainIrisDataset = Artifact(name="train_iris_dataset")
KnnModelArtifact = Artifact(name="knn_model")

# Train a knn model
@task(
    enable_deck=True,
    container_image=image,
    requests=Resources(cpu="2", mem="2Gi"),
)
def train_knn_model(
    dataset: Annotated[pd.DataFrame, TrainIrisDataset], n_neighbors: int
) -> Annotated[FlyteFile, KnnModelArtifact]:
    
    working_dir = Path(current_context().working_directory)
    model_file = working_dir / "model.joblib"
    
    X_train, y_train = dataset.drop("target", axis="columns"), dataset["target"]
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    joblib.dump(knn, model_file)

    return KnnModelArtifact.create_from(model_file)
