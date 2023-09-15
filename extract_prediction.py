from transformers import LayoutLMForTokenClassification, LayoutLMv2Processor,LayoutLMv3ForTokenClassification,LayoutLMv3Processor,AutoProcessor,AutoModelForTokenClassification
from PIL import Image, ImageDraw, ImageFont
import torch
from pytesseract import Output, pytesseract
from typing import List
from datasets import load_dataset, load_from_disk
import random

model = AutoModelForTokenClassification.from_pretrained("dwitidibyajyoti/layoutmlv3_sunday_sep3_v5")
# we'll use the Auto API here - it will load LayoutLMv3Processor behind the scenes,
# based on the checkpoint we provide from the hub
processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

def scale_bounding_box(box, width_scale : float = 1.0, height_scale : float = 1.0) -> List[int]:
    return [
        int(box['left'] * width_scale),
        int(box['top'] * height_scale),
        int(box['right'] * width_scale),
        int(box['bottom'] * height_scale)
    ]

def perform_ocr(image):
    # Perform OCR on the image using pytesseract
    ocr_results = pytesseract.image_to_data(image,output_type=pytesseract.Output.DICT, lang='eng')
    ocr_data = []
    for i in range(len(ocr_results["text"])):
        word = ocr_results["text"][i]
        bounding_box = {
            "left": ocr_results["left"][i],
            "top": ocr_results["top"][i],
            "right": ocr_results["left"][i] + ocr_results["width"][i],
            "bottom": ocr_results["top"][i] + ocr_results["height"][i]
        }

        ocr_data.append({"word": word, "bounding_box": bounding_box})

    return ocr_data


def unnormalize_box(bbox, width, height):
     return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),]

def generate_random_array(length):
    # Generate a random array of the specified length with values between 0 and 13
    random_array = [random.randint(0, 13) for _ in range(length)]
    return random_array

def iob_to_label(label):
    label = label[2:]
    if not label:
      return 'other'
    return label

def run_inference(path, model=model, processor=processor, output_image=True):
    
    image = Image.open(path).convert("RGB")
    width, height = image.size
    width_scale = 1000 / width
    height_scale = 1000 / height
    ocr_result = perform_ocr(image)  # Replace with your OCR implementation
    
    words = []
    boxes = []
    


    
    for row in ocr_result:
        boxes.append(
            scale_bounding_box(
                row["bounding_box"],
                width_scale,
                height_scale
            )
        )
        words.append(row["word"])
    word_labels = generate_random_array(len(boxes))

    encoding = processor(image, words, boxes=boxes,word_labels=word_labels, return_tensors="pt")
    # for k,v in encoding.items():
    #     print(k,v.shape)
    with torch.no_grad():
        outputs = model(**encoding)
    
    logits = outputs.logits
    logits.shape
    
    predictions = logits.argmax(-1).squeeze().tolist()
    
    labels = encoding.labels.squeeze().tolist()

    


    token_boxes = encoding.bbox.squeeze().tolist()
    width, height = image.size
    
    true_predictions = [model.config.id2label[pred] for pred, label in zip(predictions, labels) if label != - 100]
    true_labels = [model.config.id2label[label] for prediction, label in zip(predictions, labels) if label != -100]
    true_boxes = [unnormalize_box(box, width, height) for box, label in zip(token_boxes, labels) if label != -100]
    
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(image)
    
    font = ImageFont.load_default()
    

    
    # # label2color = {'question':'blue', 'answer':'green', 'header':'orange', 'other':'violet'}
    label2color = {
        "table": "blue",
        "value": "green",
        "ignore": "red",
        "key": "yellow",
        "column":"orange",'other':'violet'}
    
    
    
    # for prediction, box in zip(true_predictions, true_boxes):
    #     predicted_label = iob_to_label(prediction).lower()
    #     draw.rectangle(box, outline=label2color[predicted_label])
    #     draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill=label2color[predicted_label], font=font)
    
    prediction_list = [];
    for prediction, box in zip(true_predictions, true_boxes):
        predicted_label = iob_to_label(prediction).lower()
        if([0.0, 0.0, 0.0, 0.0] != box):
            x1,x2,x3,x4 = box
            cropped_image = image.crop([x1-2,x2-2,x3+5,x4+5])
            custom_config = '--psm 10'
            extracted_text = pytesseract.image_to_string(cropped_image,config=custom_config)
            prediction_list.append({str(predicted_label): str(extracted_text),"box":box})
        draw.rectangle([x1-2,x2-2,x3+5,x4+5], outline=label2color[predicted_label])
        draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill=label2color[predicted_label], font=font)
    
    
    # Group words into lines based on vertical position (top and bottom coordinates)
    lines = []
    current_line = []
        # Sort OCR data by vertical position (top coordinate)
    sorted_data = sorted(prediction_list, key=lambda word: word["box"][1])
    
    for word in sorted_data:
        
        if not current_line or word["box"][1] - current_line[-1]["box"][1] <= 5:
            current_line.append(word)
        else:
            lines.append(current_line)
            current_line = [word]    
    



    for i, line in enumerate(lines):
        print(line)
    
    return image
    
    
    
result_image = run_inference('./lyoutml-ml-imgage for fine-tune/TS SP3 MND MOTOR 5041_June2023 + Annexures_page-0024.jpg')
result_image.save('output_image.jpeg')  