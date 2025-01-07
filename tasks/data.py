"""
Contains the code to download the iris dataset from scikit-learn.
"""

import pandas as pd
import seaborn as sns
from containers import image
from flytekit import Deck, Resources, current_context, task
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from flytekit.core.artifact import Artifact
from typing_extensions import Annotated
import os

from tasks.utils import _convert_fig_into_html

# Define Artifact Specifications
RawIrisDataset = Artifact(name="raw_iris_dataset")
TrainIrisDataset = Artifact(name="train_iris_dataset")
TestIrisDataset = Artifact(name="test_iris_dataset")


# --------------------------------
# Download Iris dataset
# --------------------------------
@task(
    cache=True,
    cache_version="4",
    container_image=image,
    requests=Resources(cpu="2", mem="2Gi"),
)
def download_iris_dataset() -> Annotated[pd.DataFrame, RawIrisDataset]:
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df["target"] = iris.target
    return RawIrisDataset.create_from(iris_df)

# --------------------------------
# process & visualize the dataset
# --------------------------------
@task(
    enable_deck=True,
    container_image=image,
    requests=Resources(cpu="2", mem="2Gi"),
)
def process_dataset(data_df: Annotated[pd.DataFrame, RawIrisDataset]) -> tuple[
    Annotated[pd.DataFrame, TrainIrisDataset], 
    Annotated[pd.DataFrame, TestIrisDataset]
]:

    # Perform the train-test split
    train_df, test_df = train_test_split(
        data_df, test_size=0.2, random_state=42, stratify=data_df["target"]
    )

    # Seaborn pairplot full dataset
    pairplot = sns.pairplot(data_df, hue="target")

    metrics_deck = Deck("Metrics")
    metrics_deck.append(_convert_fig_into_html(pairplot))

    # Save the pairplot as image if local execution
    if "FLYTE_INTERNAL_EXECUTION_ID" not in os.environ:
        os.makedirs("reports", exist_ok=True)
        pairplot.savefig("reports/pairplot.png")
        
    # Create test & train data artifacts
    return (
        TrainIrisDataset.create_from(train_df),
        TestIrisDataset.create_from(test_df)
    )

if __name__ == "__main__":
    download_iris_dataset()
    process_dataset(download_iris_dataset())