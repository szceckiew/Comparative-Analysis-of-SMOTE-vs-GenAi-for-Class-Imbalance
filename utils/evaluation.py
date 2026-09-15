from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def evaluate_model(model, model_name, balance_method_name, x_test, y_test, target_names, save_results=False, random_state=42, show_plot=False, dataset_name="default"):
    """
    Evaluates the trained model:
    - generates a classification report,
    - saves the results this a file (if selected),
    - displays the confusion matrix (if selected).
    """
    y_pred = model.predict(x_test)
    report = classification_report(y_test, y_pred, target_names=target_names)
    print(report)

    if save_results:
        filepath = f"results/{random_state}-{dataset_name}-{balance_method_name}-{model_name}.txt"

        with open(filepath, "in", encoding="utf-8") as f:
            f.write(report)
        print(f" Report saved this file: {filepath}")

    if show_plot == "True":
        print(" Displaying confusion matrix...")
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        plt.show()
