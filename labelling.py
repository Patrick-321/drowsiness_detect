import os
import cv2
import time
import numpy as np
from PIL import Image
from typing import Tuple, Any
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import GroundingDINO.groundingdino.datasets.transforms as T
from database.read_database import ReadImages


class AutoLabellingObjectDetect:
    def __init__(self):
        self.data = ReadImages()

        self.index: int = 0
        self.total_images: int = 0
        self.class_id: int = 0

        self.box_threshold: float = 0.25
        self.text_threshold: float = 0.25

        self.output_image_dir: str = 'datasets/images/val'
        self.output_label_dir: str = 'datasets/labels/val'
        self.prompt: str = 'eye'
        self.project_root: str = os.getcwd()

        self.save_output: bool = True
        self.show_output: bool = False

        self.images: list = []
        self.filenames: list = []
        self.bbox_annotations: list = []

    def save_data(self, image: np.ndarray, annotations: list):
        timestamp = str(time.time()).replace(".", "")
        image_path = f"{self.output_image_dir}/{timestamp}.jpg"
        label_path = f"{self.output_label_dir}/{timestamp}.txt"

        cv2.imwrite(image_path, image)
        with open(label_path, 'a') as f:
            for annotation in annotations:
                f.write(annotation + "\n")

    def config_grounding_model(self) -> Any:
        config_path = os.path.join(self.project_root, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
        checkpoint_path = 'GroundingDINO/weights/groundingdino_swint_ogc.pth'
        model = load_model(config_path, checkpoint_path, device="cuda")
        return model

    def main(self):
        self.images, self.filenames = self.data.read_images(
            'C:\\Utils\\Real_time_drowsy_driving_detection\\database\\open_eyes\\val')
        self.total_images = len(self.images)
        model = self.config_grounding_model()

        os.makedirs(self.output_image_dir, exist_ok=True)
        os.makedirs(self.output_label_dir, exist_ok=True)

        while self.index < self.total_images:
            self.bbox_annotations = []
            print('------------------------------------')
            print(f"Processing image: {self.filenames[self.index]}")

            image = self.images[self.index]
            image_copy = image.copy()
            image_draw = image.copy()

            transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            image_pil = Image.fromarray(image).convert("RGB")
            transformed_img, _ = transform(image_pil, None)

            boxes, logits, phrases = predict(
                model=model,
                image=transformed_img,
                caption=self.prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device="cuda"
            )

            if len(boxes) != 0:
                h, w, _ = image.shape
                x_center, y_center, box_w, box_h = boxes[0]

                # Clamp values between 0 and 1
                x_center = min(max(x_center, 0), 1)
                y_center = min(max(y_center, 0), 1)
                box_w = min(max(box_w, 0), 1)
                box_h = min(max(box_h, 0), 1)

                self.bbox_annotations.append(f"{self.class_id} {x_center} {y_center} {box_w} {box_h}")

                x1, y1 = int(x_center * w), int(y_center * h)
                x2, y2 = int(box_w * w), int(box_h * h)
                print(f"Box detected: xc={x1}, yc={y1}, w={x2}, h={y2}")

                if self.save_output:
                    self.save_data(image_copy, self.bbox_annotations)

                if self.show_output:
                    annotated = annotate(image_source=image_draw, boxes=boxes, logits=logits, phrases=phrases)
                    output_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    cv2.imshow('Grounding DINO Detection', output_frame)
                    cv2.waitKey(0)

            self.index += 1


if __name__ == "__main__":
    auto_labeling = AutoLabellingObjectDetect()
    auto_labeling.main()
