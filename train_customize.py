import os
import matplotlib.pyplot as plt
from ultralytics import YOLO

def train_yolo(
    model_path="yolov8n.pt",
    data_yaml="dataset.yaml",
    epochs=20,
    batch_size=32,
    img_size=640,
    optimizer="Adam",
    lr0=0.001,
    project="yolo_train_project",
    name="yolov8n_custom"
):
    model = YOLO(model_path)

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        optimizer=optimizer,
        lr0=lr0,
        project=project,
        name=name,
        save=True,
        verbose=True
    )

    # Run evaluation
    metrics = model.val()

    # Extract curves
    pr_curve = metrics.box.pr_curve  # shape: (num_classes, 100)
    confs = metrics.box.conf  # shape: (100,)

    # Plot PR curve
    plt.figure()
    for cls in range(pr_curve.shape[0]):
        plt.plot(confs, pr_curve[cls], label=f"Class {cls}")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{project}/{name}/pr_curve.png")

    # Plot Recall-Confidence Curve
    plt.figure()
    for cls in range(pr_curve.shape[0]):
        recalls = metrics.box.re_curve[cls]
        plt.plot(confs, recalls, label=f"Class {cls}")
    plt.title("Recall-Confidence Curve")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Recall")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{project}/{name}/recall_conf_curve.png")

    # Plot Precision-Confidence Curve
    plt.figure()
    for cls in range(pr_curve.shape[0]):
        precs = metrics.box.precision_curve[cls]
        plt.plot(confs, precs, label=f"Class {cls}")
    plt.title("Precision-Confidence Curve")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{project}/{name}/precision_conf_curve.png")

if __name__ == "__main__":
    train_yolo()
