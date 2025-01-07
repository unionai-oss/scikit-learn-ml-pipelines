from typing import List

import numpy as np
from flytekit import Resources, task
from sklearn.neighbors import KNeighborsClassifier
from containers import image
from union.actor import ActorEnvironment
from flytekit.core.artifact import Artifact
from typing_extensions import Annotated

# --------------------------------
# batch prediction task
# --------------------------------
@task(
    container_image=image,
    enable_deck=True,
    requests=Resources(cpu="2", mem="2Gi"),
)
def batch_knn_predict(
    pred_data: List[List[float]],
    model: KNeighborsClassifier, 
) -> List[int]:
    predictions = model.predict(pred_data)
    return predictions.tolist()

# --------------------------------
# near real-time prediction task with actors
# --------------------------------

actor = ActorEnvironment(
    name="my-actor",
    container_image=image,
    replica_count=1,
    ttl_seconds=120,
    requests=Resources(
        cpu="2",
        mem="500Mi",
    ),
)

@actor.task
def actor_knn_predict(
    model: KNeighborsClassifier, pred_data: List[List[float]]
) -> List[int]:
    predictions = model.predict(pred_data)
    return predictions.tolist()

@actor.task
def actor_model_predict(
    model: KNeighborsClassifier, pred_data: List[List[float]]
) -> List[int]:
    predictions = model.predict(pred_data)
    return predictions.tolist()