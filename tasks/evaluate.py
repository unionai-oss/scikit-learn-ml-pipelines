import html
from textwrap import dedent

import matplotlib.pyplot as plt
import pandas as pd
from flytekit import Deck, current_context
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.neighbors import KNeighborsClassifier
from flytekit import Deck, Resources, task
from containers import image
import os
from flytekit.core.context_manager import FlyteContextManager
from flytekit import Artifact
from typing_extensions import Annotated

TestIrisDataset = Artifact(name="test_iris_dataset")

from tasks.utils import _convert_fig_into_html


@task(
    enable_deck=True,
    container_image=image,
    requests=Resources(cpu="2", mem="2Gi"),
)
def evaluate_model(
    model: KNeighborsClassifier, dataset: Annotated[pd.DataFrame, TestIrisDataset]
) -> KNeighborsClassifier:
    ctx = current_context()

    X_test, y_test = dataset.drop("target", axis="columns"), dataset["target"]
    y_pred = model.predict(X_test)

    # Plot confusion matrix in deck
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)

    metrics_deck = Deck("Metrics")
    metrics_deck.append(_convert_fig_into_html(fig))

    # Add classification report
    report = html.escape(classification_report(y_test, y_pred))
    html_report = dedent(
        f"""\
    <h2>Classification report</h2>
    <pre>{report}</pre>"""
    )
    metrics_deck.append(html_report)
    ctx.decks.insert(0, metrics_deck)
    
    # save the classification report locally
    flctx = FlyteContextManager.current_context()
    if flctx.execution_state.is_local_execution():
        os.makedirs("reports", exist_ok=True)
        with open("reports/classification_report.html", "w") as f:
            f.write(html_report)

    return model
