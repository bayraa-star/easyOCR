import torch
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from model import Model
from utils import CTCLabelConverter, AttrDict
import yaml
import cv2


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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Model(self.opt).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        if "module." in list(state_dict.keys())[0]:
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.converter = CTCLabelConverter(self.opt.character)

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

        preds_size = torch.IntTensor([preds.size(1)])
        _, preds_index = preds.max(2)
        pred_str = self.converter.decode_greedy(preds_index[0].cpu(), preds_size.cpu())

        if "[blank]" in pred_str:
            pred_str = pred_str.replace("[blank]", "")
            cleaned = []
            for char in pred_str:
                if not cleaned or char != cleaned[-1]:
                    cleaned.append(char)
            pred_str = "".join(cleaned)

        return pred_str
