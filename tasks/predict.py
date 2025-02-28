from typing import List

import numpy as np
from flytekit import Resources, task, FlyteFile
from sklearn.neighbors import KNeighborsClassifier
from containers import image, actor
from union.actor import ActorEnvironment
from flytekit.core.artifact import Artifact
from typing_extensions import Annotated
import union
import joblib

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
    model: FlyteFile, 
) -> List[int]:
    
    model = joblib.load(model.download())
    predictions = model.predict(pred_data)
    return predictions.tolist()

# --------------------------------
# near real-time prediction task with actors
# --------------------------------

# @actor.task
# def actor_knn_predict(
#     model: FlyteFile, pred_data: List[List[float]]
# ) -> List[int]:
#     predictions = model.predict(pred_data)
#     return predictions.tolist()


@union.actor_cache
def load_model(model: FlyteFile) -> KNeighborsClassifier:
    # actor caching is useful for large models
    model = joblib.load(model.download())
    
    return model

@actor.task
def actor_model_predict(
    model: FlyteFile, pred_data: List[List[float]]
) -> List[int]:
    loaded_model = load_model(model)
    predictions = loaded_model.predict(pred_data)
    return predictions.tolist()