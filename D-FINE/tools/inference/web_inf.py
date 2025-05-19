import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.core import YAMLConfig

CLASS_NAMES = ["Flying-object", "Flying-object", "Non-flying-object"]  # <-- Replace with your actual classes


def draw(image_pil, labels, boxes, scores, thrh=0.4):
    draw = ImageDraw.Draw(image_pil)
    scr = scores[0]
    lab = labels[0][scr > thrh]
    box = boxes[0][scr > thrh]
    scrs = scr[scr > thrh]

    for j, b in enumerate(box):
        label_idx = lab[j].item()
        class_name = CLASS_NAMES[label_idx] if label_idx < len(CLASS_NAMES) else f"Class {label_idx}"
        confidence = round(scrs[j].item(), 2)
        label_text = f"{class_name} {confidence}"

        draw.rectangle(list(b), outline="red", width=3)
        draw.text((b[0], b[1]), text=label_text, fill="blue")



def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, required=True)
    parser.add_argument("-d", "--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg = YAMLConfig(args.config, resume=args.resume)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]
    else:
        raise AttributeError("Only support resume to load model.state_dict by now.")

    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    device = args.device
    model = Model().to(device)
    model.eval()

    # Setup OpenCV webcam capture
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    transforms = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
    )

    print("Starting live webcam inference. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert to PIL Image for processing
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        w, h = frame_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)

        im_data = transforms(frame_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            labels, boxes, scores = model(im_data, orig_size)

        draw(frame_pil, labels, boxes, scores)

        # Convert back to OpenCV image and show
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        cv2.imshow("D-FINE Live Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()