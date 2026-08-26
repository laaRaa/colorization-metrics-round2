import os

import matplotlib.pyplot as plt
from dataset_analysis import ALTERATIONS_WITH_REF, get_dataset_information
from dataset_utils import ALTERATIONS, METRICS, DatasetAvailables, ResultsManager


def plot_image_metrics(
    image_name: str,
    dataset: str = DatasetAvailables.UNSPLASH.value,
    graph_title: bool = False,
) -> None:
    """Plot and save image metrics for altered images.

    This function generates plots for various metrics of altered images
        in a specified dataset.
    The plots are saved as both PNG and PDF files in a designated output directory.

    Args:
        image_name: The file name of the image.
        dataset: The name of the dataset to plot metrics for.
        graph_title: Whether to display the graph title.
    """
    results = ResultsManager(dataset)
    _, _, _, _, output_dir = get_dataset_information(results)
    output_dir = os.path.join(output_dir, image_name)

    for metric_name, metric in METRICS.items():
        # Check if already drawn
        pdf_plot_path = os.path.join(output_dir, f"{image_name}_{metric_name}.pdf")
        if os.path.exists(pdf_plot_path):
            continue

        metric_result = results.get_sub_metric_dataframe(metric_name)
        metric_result = metric_result[metric_result["Image"] == image_name + ".jpg"]
        greater_is_better = metric.higher_is_better
        alteration_dict = ALTERATIONS_WITH_REF if metric.uses_reference else ALTERATIONS

        # Sort the altered images based on the metric value
        values = metric_result.sort_values(by="Value", ascending=not greater_is_better)
        sorted_alterations = values["Alteration"].tolist()
        alteration_title = [
            alteration_dict[alteration_name].name
            for alteration_name in sorted_alterations
            if alteration_name in alteration_dict
        ]
        alteration_color = {
            alteration_dict[alteration_name].name: alteration_dict[
                alteration_name
            ].type_color
            for alteration_name in sorted_alterations
            if alteration_name in alteration_dict
        }

        metric_values = [
            metric_result[metric_result["Alteration"] == alteration]["Value"].values[0]
            for alteration in sorted_alterations
            if alteration in alteration_dict
        ]
        if len(metric_values) == 0:
            print(f"Not on {image_name} with {metric_name}.")
            continue

        plt.figure(figsize=(10, 6))
        plt.plot(alteration_title, metric_values, "o-")

        # Add value on points
        offset = max(metric_values) / 100
        for i, value in enumerate(metric_values):
            plt.text(
                i, value + offset, f"{value:.2f}", ha="center", va="bottom", fontsize=9
            )

        # Customize x-tick labels and add color coding
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.xlabel("Altered Image Filename")
        for label in plt.gca().get_xticklabels():
            text = label.get_text()
            if text in alteration_color:
                label.set_color(alteration_color[text])

        # Customize y-labels
        plt.ylabel(metric_name)
        if graph_title:
            plt.title(f"{metric_name} for all altered images")
        plt.tight_layout()
        plt.grid(True)

        # Save the plot
        pdf_plot_path = os.path.join(output_dir, f"{image_name}_{metric_name}.pdf")
        plt.savefig(pdf_plot_path)
        plt.close()


if __name__ == "__main__":
    RESULTS = ResultsManager(DatasetAvailables.UNSPLASH.value)
    _, _, _, _, OUTPUT_DIR = get_dataset_information(RESULTS)
    del RESULTS
    directories = [
        entry
        for entry in os.listdir(OUTPUT_DIR)
        if os.path.isdir(os.path.join(OUTPUT_DIR, entry))
    ]
    directories.sort()
    for directory in directories:
        print(f"--- {directory}")
        plot_image_metrics(directory)
