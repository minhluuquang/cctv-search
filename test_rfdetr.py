import numpy as np
import supervision as sv
from PIL import Image
from rfdetr.detr import RFDETRMedium
from rfdetr.util.coco_classes import COCO_CLASSES

model = RFDETRMedium(resolution=1024)


image = Image.open("test_images/test.png")
image_np = np.array(image)
detections = model.predict(image, threshold=0.5)

# Handle both single and list of detections
if isinstance(detections, list):
    detections = detections[0]

labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]

annotated_image = sv.BoxAnnotator().annotate(image_np, detections)
annotated_image = sv.LabelAnnotator().annotate(
    annotated_image, detections, labels)

# Convert numpy array back to PIL Image for saving
annotated_pil = Image.fromarray(annotated_image)
annotated_pil.save("test_output.png")
print(f"Detected {len(detections)} objects")
print(f"Classes: {labels}")
print("Saved to test_output.png")
