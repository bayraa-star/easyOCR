import torch
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from model import Model
from utils import CTCLabelConverter, AttrDict
import yaml
import cv2
import torch.nn as nn
import math


def load_config(file_path):
    """Load the configuration from a YAML file and prepare the character set."""
    with open(file_path, "r", encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    opt.character = opt.number + opt.symbol + opt.lang_char
    return opt


def resize_with_pad(img, target_height, target_width):
    """Resize the image while maintaining aspect ratio and pad to target size."""
    w, h = img.size
    ratio = target_height / h
    new_w = int(w * ratio)
    if new_w > target_width:
        ratio = target_width / w
        new_h = int(h * ratio)
        img = img.resize((target_width, new_h), Image.BICUBIC)
        padding = (0, 0, 0, target_height - new_h)  # Pad bottom
        img = ImageOps.expand(img, padding, fill=0)  # Black padding
    else:
        img = img.resize((new_w, target_height), Image.BICUBIC)
        padding = (0, 0, target_width - new_w, 0)  # Pad right
        img = ImageOps.expand(img, padding, fill=0)  # Black padding
    return img


class LicensePlateRecognizer:
    def __init__(self, model_path, config_path):
        """Initialize the recognizer with model and config paths."""
        self.opt = load_config(config_path)
        if "CTC" in self.opt.Prediction:
            self.opt.num_class = len(self.opt.character) + 1  # +1 for CTC blank
        else:
            self.opt.num_class = len(self.opt.character)
        self.device = torch.device("cpu")
        self.model = Model(self.opt).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        if "module." in list(state_dict.keys())[0]:
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.converter = CTCLabelConverter(self.opt.character)
        self.blank_id = 0  # Assuming blank is index 0
        self.char_list = list(self.opt.character)  # Chars at indices 1 to len(character)
        self.conf_threshold = 0.7  # Threshold for keeping characters

    def predict(self, image_array):
        """Predict text from a NumPy image array."""
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        img = Image.fromarray(gray)  # Convert to PIL Image
        img = resize_with_pad(img, self.opt.imgH, self.opt.imgW)
        img = transforms.ToTensor()(img)  # To tensor [0, 1]
        img = transforms.Normalize(mean=[0.5], std=[0.5])(img)  # Normalize [-1, 1]
        img = img.unsqueeze(0).to(self.device)  # Add batch dimension

        dummy_text = torch.zeros(1, self.opt.batch_max_length, dtype=torch.long).to(
            self.device
        )

        with torch.no_grad():
            preds = self.model(img, dummy_text).log_softmax(2)
            probs = preds.exp()

        preds_size = torch.IntTensor([preds.size(1)])
        _, preds_index = preds.max(2)

        # Use converter for pred_str (assuming it returns list for batch=1)
        raw_pred_str = self.converter.decode_greedy(preds_index[0].cpu(), preds_size.cpu())
        pred_str = raw_pred_str[0] if isinstance(raw_pred_str, list) else raw_pred_str

        # Manual cleaning if needed (fallback, but our conf logic handles CTC properly)
        if "[blank]" in pred_str:
            pred_str = pred_str.replace("[blank]", "")
            cleaned = []
            for char in pred_str:
                if not cleaned or char != cleaned[-1]:
                    cleaned.append(char)
            pred_str = "".join(cleaned)

        # Collect per-character confidences with proper CTC decoding logic
        length = preds_size.item()
        preds_index_list = preds_index[0].cpu().tolist()  # (T,) as list
        orig_times = [idx for idx, label in enumerate(preds_index_list) if label != self.blank_id]

        char_confs = []
        if orig_times:
            non_blank_labels = [preds_index_list[t] for t in orig_times]
            j = 0
            while j < len(non_blank_labels):
                curr = non_blank_labels[j]
                group_start = j
                while j < len(non_blank_labels) - 1 and non_blank_labels[j + 1] == curr:
                    j += 1
                group_end = j
                group_times = orig_times[group_start:group_end + 1]
                group_confs = [probs[0, t, curr].item() for t in group_times]
                max_conf = max(group_confs)
                char_confs.append(max_conf)
                j += 1

        # Ensure lengths match (in case of manual cleaning discrepancies)
        if len(char_confs) != len(pred_str):
            # Fallback: average or pad with 1.0, but log warning
            import logging
            logging.warning(f"Length mismatch: len(char_confs)={len(char_confs)}, len(pred_str)={len(pred_str)}. Using average conf.")
            avg_conf = sum(char_confs) / len(char_confs) if char_confs else 0.0
            char_confs = [avg_conf] * len(pred_str)

        # Filter characters and confidences based on threshold
        filtered_chars = []
        filtered_confs = []
        for char, conf in zip(pred_str, char_confs):
            if conf >= self.conf_threshold:
                filtered_chars.append(char)
                filtered_confs.append(conf)

        filtered_pred_str = "".join(filtered_chars)
        char_details = list(zip(filtered_chars, filtered_confs))

        # Compute overall confidence as geometric mean of filtered confidences
        if filtered_confs:
            log_confs = [math.log(max(c, 1e-10)) for c in filtered_confs]
            avg_log = sum(log_confs) / len(log_confs)
            confidence = math.exp(avg_log)
        else:
            confidence = 0.0

        return filtered_pred_str, confidence, char_details