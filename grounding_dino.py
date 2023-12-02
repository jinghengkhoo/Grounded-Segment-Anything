import torch
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        color = (252, 3, 3)
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask

def convert_to_output(tgt):
    im_height, im_width = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]

    res = []
    objectId = 0
    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([im_width, im_height, im_width, im_height])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]

        class_name, confidence = label.replace(")", "").split("(")

        obj = {
            "confidence": float(confidence),
            "class": class_name,
            "boundingBox": {
                "top": int(box[1]),
                "left": int(box[0]),
                "width": int(box[2]-box[0]),
                "height": int(box[3]-box[1])
            },
            "objectId": str(objectId)
        }

        objectId += 1

        res.append(obj)

    return res


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device="cpu"):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def check_if_roi(boxes_filt, pred_phrases, size):

    pred_phrases = np.array(pred_phrases)

    roi = {
        "x0": 218,
        "y0": 128,
        "x1": 595,
        "y1": 463
    }

    H, W = size[1], size[0]

    inds_to_keep = np.array([])

    for idx, box in enumerate(boxes_filt):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        print(x0, y0, x1, y1)
        
        # if rectangle has area 0, no overlap
        if x0 == x1 or y0 == y1 or roi["x1"] == roi["x0"] or roi["y0"] == roi["y1"]:
            continue
        
        # If one rectangle is on left side of other
        if x0 > roi["x1"] or roi["x0"] > x1:
            print("out of roi x")
            continue

        # If one rectangle is above other
        if y1 > roi["y0"] or roi["y1"] > y0:
            print("out of roi y")
            continue

        np.array.append(inds_to_keep, idx)

    if inds_to_keep:
        return boxes_filt[inds_to_keep], pred_phrases[inds_to_keep]
    else:
        return [], []


def run_dino(model, image_path, text_prompt, semb=False):
    box_threshold = 0.4
    text_threshold = 0.4
    device = "cuda"

    # load image
    image_pil, image = load_image(image_path)

    # run model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold, device=device
    )

    size = image_pil.size

    
    boxes_filt, pred_phrases = check_if_roi(boxes_filt, pred_phrases, size)
    #semb
    # pred_phrases_cp = pred_phrases[:]
    # possible_labels = text_prompt.split(" . ")
    # pred_phrases = []
    # for label in pred_phrases_cp:
    #     name = label.split("(")[0]
    #     if name in possible_labels:
    #         pred_phrases.append(label)

    if semb:
        pred_phrases_cp = pred_phrases[:]
        semb_materials = {
            "cardboard": "corrugated",
            "plastic bottle": "P1 PET bottle",
            "plastic cup": "other plastics",
            "white plastic bottles": "P2 HDPE bottles"
        }
        for idx, phrase in enumerate(pred_phrases_cp):
            name = phrase.split("(")[0]
            if name in semb_materials:
                pred_phrases[idx] = phrase.replace(name, semb_materials[name])
                print(f"changed {phrase} to {pred_phrases[idx]}")

    # visualize pred
    pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],  # H,W
        "labels": pred_phrases,
    }

    image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
    image_with_box.save(os.path.join("static", "grounding_dino_output.jpg"))

    return convert_to_output(pred_dict)

if __name__ == "__main__":
    config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"  # change the path of the model config file
    grounded_checkpoint = "groundingdino_swint_ogc.pth" # change the path of the model
    device = "cuda"
    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)
    print(run_dino(model, "assets/demo1.jpg", "bear"))