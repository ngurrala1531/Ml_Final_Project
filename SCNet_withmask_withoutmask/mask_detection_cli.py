import os
import argparse
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from scnet import scnet50
import torch.nn as nn
import face_recognition


def load_model(model_path):
    model = scnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)  
    return input_batch, image

def predict_with_labels(model, input_tensor, class_labels):
    with torch.no_grad():
        model.eval()
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        class_label = class_labels[predicted_class]
    return class_label, probabilities

def overlay_label_and_bounding_box(image, face_locations, predicted_label):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for face_location in face_locations:
        top, right, bottom, left = face_location
        
        draw.rectangle([left, top, right, bottom], outline="red", width=2)
        
        draw.text((right + 10, top), f'Predicted: {predicted_label}', fill="white", font=font)

    return image

def detect_faces(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    return face_locations

def process_input(input_path, model, class_labels):
    if os.path.isfile(input_path):  # If input is a file, process the single image
        input_tensor, original_image = preprocess_image(input_path)
        face_locations = detect_faces(input_path)
        predicted_label, probabilities = predict_with_labels(model, input_tensor, class_labels)
        output_image = overlay_label_and_bounding_box(original_image, face_locations, predicted_label)
        output_path = f'output_{os.path.basename(input_path)}'
        output_image.save(output_path)
        print(f'Output saved to {output_path}')
    elif os.path.isdir(input_path): 
        output_folder = 'outputs'
        os.makedirs(output_folder, exist_ok=True)
        for filename in os.listdir(input_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(input_path, filename)
                input_tensor, original_image = preprocess_image(image_path)
                face_locations = detect_faces(image_path)
                predicted_label, probabilities = predict_with_labels(model, input_tensor, class_labels)
                output_image = overlay_label_and_bounding_box(original_image, face_locations, predicted_label)
                output_path = os.path.join(output_folder, f'output_{filename}')
                output_image.save(output_path)
                print(f'Output saved to {output_path}')
    else:
        print(f'Error: The provided path {input_path} is neither a file nor a folder.')

def main():
    parser = argparse.ArgumentParser(description='Mask Detection CLI for Image or Folder with Face Detection')
    parser.add_argument('input_path', type=str, help='Path to the input image or folder')
    parser.add_argument('model_path', type=str, help='Path to the trained model')
    args = parser.parse_args()


    class_labels = ["with_mask", "without_mask"]

    model = load_model(args.model_path)

    process_input(args.input_path, model, class_labels)

if __name__ == '__main__':
    main()
