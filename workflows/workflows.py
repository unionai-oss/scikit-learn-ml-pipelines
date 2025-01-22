# --------------------------------
# import modules and libraries
# -------------------------------- 
from tasks.data import download_iris_dataset, process_dataset
from tasks.train import train_knn_model
from tasks.evaluate import evaluate_model
from tasks.predict import batch_knn_predict, actor_knn_predict, actor_model_predict
from flytekit.core.artifact import Artifact
from typing_extensions import Annotated
from sklearn.neighbors import KNeighborsClassifier
from typing import List
from flytekit import task, workflow, Resources, ImageSpec

# artifact to query between workflows
KnnModelArtifact = Artifact(name="knn_model")


# --------------------------------
# Training Workflow (could be seperate workflows for data)
# --------------------------------
@workflow
def train_iris_classification(n_neighbors: int=3, 
                              pred_data: List[List[float]]=[[1.2, 2.1, 3.3, 4.0]]) -> None:
    data = download_iris_dataset()
    train, test = process_dataset(data)
    model = train_knn_model(dataset=train, n_neighbors=n_neighbors)
    evaluate_model(model, test)
    batch_knn_predict(model=model, pred_data=pred_data)

    return model
    
# union run --remote workflows/workflows.py train_iris_classification

# --------------------------------
# Batch prediction workflow
# --------------------------------
@workflow
def batch_prediction_knn(
    model: KNeighborsClassifier = KnnModelArtifact.query(),
    pred_data: List[List[float]] = [[1.2, 2.1, 3.3, 4.0], 
                                    [5.2, 6.3, 7.1, 8.3], 
                                    [9.1, 1.0, 1.1, 1.2]]
) -> list[int]:
    pred = batch_knn_predict(
        pred_data=pred_data,
        model=model
    )
    return pred
#union run --remote workflows/workflows.py batch_prediction_knn

# --------------------------------
# Near real-time prediction workflow with Actors
# --------------------------------
@workflow
def actor_prediction_knn(
    model: KNeighborsClassifier = KnnModelArtifact.query(),
    pred_data: List[List[float]] = [[1.2, 2.1, 3.3, 4.0], 
                                    [5.2, 6.3, 7.1, 8.3], 
                                    [9.1, 1.0, 1.1, 1.2]]
) -> list[int]:
    pred = actor_knn_predict(
        pred_data=pred_data,
        model=model
    )
    return pred

#union run --remote workflows/workflows.py actor_prediction_knn
