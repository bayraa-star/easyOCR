import torch
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from model import Model
from utils import CTCLabelConverter, AttrDict
import yaml

def load_config(file_path):
    """Load the configuration from a YAML file and prepare the character set."""
    with open(file_path, 'r', encoding="utf8") as stream:
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
        # Scale to fit width if new width exceeds target
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

def predict_plate(image_path, model_path, config_path):
    """Predict the text on a license plate image using the trained OCR model."""
    # Load configuration
    opt = load_config(config_path)

    # Compute num_class
    if 'CTC' in opt.Prediction:
        opt.num_class = len(opt.character) + 1  # +1 for CTC blank token
    else:
        opt.num_class = len(opt.character)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = Model(opt).to(device)
    state_dict = torch.load(model_path, map_location=device)
    if 'module.' in list(state_dict.keys())[0]:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # Load and preprocess the image
    img = Image.open(image_path).convert('L')  # Grayscale
    img = resize_with_pad(img, opt.imgH, opt.imgW)  # Resize with padding
    img = transforms.ToTensor()(img)  # To tensor [0, 1]
    img = transforms.Normalize(mean=[0.5], std=[0.5])(img)  # Normalize [-1, 1]
    img = img.unsqueeze(0).to(device)  # Add batch dimension

    # Dummy text tensor (required by model but not used in prediction)
    dummy_text = torch.zeros(1, opt.batch_max_length, dtype=torch.long).to(device)

    # Run inference
    with torch.no_grad():
        preds = model(img, dummy_text).log_softmax(2)  # (batch, time_steps, num_class)

    # Decode the prediction
    converter = CTCLabelConverter(opt.character)
    preds_size = torch.IntTensor([preds.size(1)])  # Sequence length
    _, preds_index = preds.max(2)  # Max probability indices
    pred_str = converter.decode_greedy(preds_index[0].cpu(), preds_size.cpu())

    # Post-process if '[blank]' appears (workaround if decode_greedy is buggy)
    if '[blank]' in pred_str:
        pred_str = pred_str.replace('[blank]', '')
        # Collapse duplicates (if not handled by decode_greedy)
        cleaned = []
        for char in pred_str:
            if not cleaned or char != cleaned[-1]:
                cleaned.append(char)
        pred_str = ''.join(cleaned)

    return pred_str

if __name__ == "__main__":
    image_path = './7500УНТ_plate_0.jpg'
    model_path = './saved_models/mn_filtered_v16-1200image/best_accuracy.pth'
    config_path = './config_files/mn_filtered_config_v6m1000.yaml'
    predicted_text = predict_plate(image_path, model_path, config_path)
    print(f"Predicted License Plate: {predicted_text}")