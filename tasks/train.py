import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from flytekit.core.artifact import Artifact
from typing_extensions import Annotated
from flytekit import Deck, Resources, task
from containers import image

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
    dataset: Annotated[pd.DataFrame, TrainIrisDataset], n_neighbors: int = 3
) -> Annotated[KNeighborsClassifier, KnnModelArtifact]:
    
    X_train, y_train = dataset.drop("target", axis="columns"), dataset["target"]
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)

    return KnnModelArtifact.create_from(model)
