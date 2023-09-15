# from datasets import load_dataset, load_from_disk
# from IPython.display import display
# from transformers import AutoModelForTokenClassification, LayoutLMv3ForTokenClassification
# from transformers import AutoProcessor
# from datasets.features import ClassLabel
# import cv2
# import numpy as np
# from PIL import ImageFont, ImageDraw
# import torch
# from pytesseract import Output, pytesseract

# # dataset = load_dataset(dataset_id)
# dataset = load_from_disk('./raw_data_finetune_layoutml3')

# processor = AutoProcessor.from_pretrained("dwitidibyajyoti/test", apply_ocr=False)


# model = AutoModelForTokenClassification.from_pretrained("dwitidibyajyoti/test")



# from PIL import Image, ImageDraw, ImageFont

# model = LayoutLMv3ForTokenClassification.from_pretrained("dwitidibyajyoti/test")

# labels = ['I-TABLE', 'I-VALUE', 'I-IGNORE', 'E-TABLE', 'O', 'B-VALUE', 'B-TABLE', 'B-IGNORE', 'S-VALUE', 'S-IGNORE', 'E-VALUE', 'S-KEY', 'E-IGNORE']
# id2label = {v: k for v, k in enumerate(labels)}
# label2color = {
#     "I-TABLE": "blue",
#     "I-VALUE": "green",
#     "I-IGNORE": "red",
#     "E-TABLE": "blue",
#     "B-VALUE": "green",
#     "B-TABLE": "blue",
#     "B-IGNORE": "red",
#     "S-VALUE": "green",
#     "S-IGNORE": "red",
#     "E-VALUE": "green",
#     "S-KEY": "yellow",
#     "E-IGNORE": "red",

# }
# query_index = 2
# query = dataset['test'][query_index]
# # print(query2)
# # print('query-56')

# # image = Image.open(query['image'])
# image = query['image'].convert("RGB")

# from typing import List
# def scale_bounding_box(box, width_scale : float = 1.0, height_scale : float = 1.0) -> List[int]:
#     return [
#         int(box['left'] * width_scale),
#         int(box['top'] * height_scale),
#         int(box['width'] * width_scale),
#         int(box['height'] * height_scale)
#     ]

# def perform_ocr(image):
#     # Perform OCR on the image using pytesseract
#     ocr_results = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
#     ocr_data = []
#     for i in range(len(ocr_results["text"])):
#         word = ocr_results["text"][i]
#         bounding_box = {
#             "left": ocr_results["left"][i],
#             "top": ocr_results["top"][i],
#             "width": ocr_results["width"][i],
#             "height": ocr_results["height"][i]
#         }
#         ocr_data.append({"word": word, "bounding_box": bounding_box})

#     return ocr_data

# # with Image.open('./lyoutml-ml-imgage for fine-tune/TS SP3 MND MOTOR 5041_June2023 + Annexures_page-0020.jpg').convert("RGB") as image:
    
# #     width, height = image.size
# #     width_scale = 1000 / width
# #     height_scale = 1000 / height

# #         # Perform OCR on the image
# #     ocr_result = perform_ocr(image)  # Replace with your OCR implementation

# #     words = []
# #     boxes = []
# #     for row in ocr_result:
# #         boxes.append(
# #             scale_bounding_box(
# #                 row["bounding_box"],
# #                 width_scale,
# #                 height_scale
# #             )
# #         )
# #         words.append(row["word"])
# # print(query)
# # image = Image.open('./lyoutml-ml-imgage for fine-tune/TS SP3 MND MOTOR 5041_June2023 + Annexures_page-0020.jpg').convert("RGB")

# # print(boxes)



# image_path = './lyoutml-ml-imgage for fine-tune/TS SP3 MND MOTOR 5041_June2023 + Annexures_page-0020.jpg'
# ocr_data = perform_ocr(image_path)
# image = Image.open(image_path).convert("RGB")

# words = []
# boxes = []
# for row in ocr_data:
#     boxes.append(
#         scale_bounding_box(
#             row["bounding_box"],
#             1,
#             1
#         )
#     )
#     words.append(row["word"])

# encoded_inputs = processor(
#     image, words, boxes=boxes, word_labels=query['ner_tags'],
#     padding="max_length", truncation=False, return_tensors="pt"
# )
# outputs = model(**encoded_inputs)


# def unnormalize_box(bbox, width, height):
#      return [
#          width * (bbox[0] / 1000),
#          height * (bbox[1] / 1000),
#          width * (bbox[2] / 1000),
#          height * (bbox[3] / 1000),
#      ]

# predictions = outputs.logits.argmax(-1).squeeze().tolist()
# token_boxes = encoded_inputs.bbox.squeeze().tolist()

# width, height = image.size

# true_predictions = [id2label[prediction] for prediction in predictions]
# true_boxes = [unnormalize_box(box, width, height) for box in token_boxes]
# print(true_boxes)
# print(true_predictions)


# # Example usage:
# image_path = './lyoutml-ml-imgage for fine-tune/TS SP3 MND MOTOR 5041_June2023 + Annexures_page-0020.jpg'
# words, bounding_boxes = get_bounding_boxes_and_words(image_path)


# print(words,'words')
# print(bounding_boxes,'bounding_box')



# # print(true_boxes)
# draw = ImageDraw.Draw(image)
# image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


# font = ImageFont.load_default()

# def iob_to_label(label):
#     label = label[2:]
#     if not label:
#         return 'other'
#     return label

# label2color = {
#     "table": "blue",
#     "key": "yellow",
#     "ignore": "red",
#     "other": "red",
#     "value":"green"
# }
# # print(true_boxes)
# for prediction, box in zip(true_predictions, true_boxes):
#     predicted_label = iob_to_label(prediction).lower()
#     # print(box)
#     # print(box,'from test-data')
#     if([0.0, 0.0, 0.0, 0.0] != box):
#         cropped_image = image.crop(box)
#         extracted_text = pytesseract.image_to_string(cropped_image)
#         # if(extracted_text != ''):
#         #     # print(prediction)
#         #     # print(extracted_text)
            
#     draw.rectangle(box, outline=label2color[predicted_label])

#     draw.text(
#         (box[0]+10, box[1]-10), text=predicted_label, fill=label2color[predicted_label], font=font
#     )


# # def predict_document_image(image_path, model, processor):
# #     # Load the image
# #     with Image.open(image_path).convert("RGB") as image:
# #         width, height = image.size
# #         width_scale = 1000 / width
# #         height_scale = 1000 / height

# #         # Perform OCR on the image
# #         ocr_result = perform_ocr(image)  # Replace with your OCR implementation
# #         words = []
# #         boxes = []
# #         for row in ocr_result:
# #             boxes.append(
# #                 scale_bounding_box(
# #                     row["bounding_box"],
# #                     width_scale,
# #                     height_scale
# #                 )
# #             )
# #             words.append(row["word"])

# #         encoding = processor(
# #             image,
# #             words,
# #             boxes=boxes,
# #             max_length=512,
# #             padding="max_length",
# #             truncation=True,
# #             return_tensors="pt"
# #         )

# #     with torch.inference_mode():
# #         output = model(
# #             input_ids=encoding["input_ids"],
# #             attention_mask=encoding["attention_mask"],
# #             bbox=encoding["bbox"],
# #             pixel_values=encoding["pixel_values"]
# #         )

# #     predictions = output.logits.argmax(-1).squeeze().tolist()
# #     token_boxes = encoding.bbox.squeeze().tolist()
    
    
# #     true_predictions = [id2label[prediction] for prediction in predictions]
# #     true_boxes = token_boxes
# #     # print(true_predictions)
# #     # print(true_boxes)
# #     count =0;
# #     for prediction, box in zip(true_predictions, true_boxes):
        
# #         predicted_label = iob_to_label(prediction).lower()

# #         if([0.0, 0.0, 0.0, 0.0] != box):
# #             print(box,'from-photo')
# #             x, y, width, height = box
# #             print(box)

# #             left = x
# #             upper = y
# #             right = x + width
# #             lower = y + height
# #             bonding_box = [left, upper, right, lower]
# #             cropped_image = image.crop(bonding_box)
# #             cropped_image.save('new_image'+str(count)+'.jpg')
# #             count+=1
# #             # break
# #             # extracted_text = pytesseract.image_to_string(cropped_image)
# #             # print(extracted_text,'coolk')
            
# #             # if(extracted_text != ''):
# #             #     print(prediction)
# #             #     print(extracted_text)


# #     return 'print'



# # predict_document_image('./lyoutml-ml-imgage for fine-tune/TS SP3 MND MOTOR 5041_June2023 + Annexures_page-0020.jpg', model, processor)




from transformers import LayoutLMForTokenClassification, LayoutLMv2Processor,LayoutLMv3ForTokenClassification,LayoutLMv3Processor,AutoProcessor
from PIL import Image, ImageDraw, ImageFont
import torch
from pytesseract import Output, pytesseract
from typing import List
from datasets import load_dataset, load_from_disk
# # load model and processor from huggingface hub
# model = LayoutLMForTokenClassification.from_pretrained("dwitidibyajyoti/layoutlm-funsd")
# # processor = LayoutLMv3ForTokenClassification.from_pretrained("dwitidibyajyoti/testv4")
# processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")

model = LayoutLMv3ForTokenClassification.from_pretrained("dwitidibyajyoti/fine_tune_layoutmlv3_model")
# processor = LayoutLMv3Processor.from_pretrained("")
processor = AutoProcessor.from_pretrained("dwitidibyajyoti/fine_tune_layoutmlv3_model", apply_ocr=False)

# helper function to unnormalize bboxes for drawing onto the image
def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]

def perform_ocr(image):
    # Perform OCR on the image using pytesseract
    ocr_results = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    ocr_data = []
    for i in range(len(ocr_results["text"])):
        word = ocr_results["text"][i]
        bounding_box = {
            "left": ocr_results["left"][i],
            "top": ocr_results["top"][i],
            "width": ocr_results["width"][i],
            "height": ocr_results["height"][i]
        }
        ocr_data.append({"word": word, "bounding_box": bounding_box})

    return ocr_data

# Define the label-color mapping for specific labels
label2color = {
    "I-TABLE": "blue",
    "I-VALUE": "green",
    "I-IGNORE": "red",
    "E-TABLE": "blue",
    "B-VALUE": "green",
    "B-TABLE": "blue",
    "B-IGNORE": "red",
    "S-VALUE": "green",
    "S-IGNORE": "red",
    "E-VALUE": "green",
    "S-KEY": "yellow",
    "E-IGNORE": "red",
    "E-COLUMN":"black",
    "S-COLUMN":"black",
    "I-COLUMN":"black",
    "B-COLUMN":"black"
}

# ['S-VALUE', 'I-VALUE', 'E-IGNORE', 'S-KEY', 'E-COLUMN', 'B-IGNORE', 'S-COLUMN', 'I-COLUMN', 'O', 'E-VALUE', 'B-COLUMN', 'S-IGNORE', 'I-IGNORE', 'B-VALUE']

# draw results onto the image
def draw_boxes(image, boxes, predictions):
    width, height = image.size
    normalizes_boxes = [unnormalize_box(box, width, height) for box in boxes]

    # draw predictions over the image
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for prediction, box in zip(predictions, normalizes_boxes):
        
        if prediction == "O":
            continue
        draw.rectangle(box, outline="black")
        # print(predictions)
        draw.rectangle(box, outline=label2color[prediction])
        draw.text((box[0] + 10, box[1] - 10), text=prediction, fill=label2color[prediction], font=font)
    return image

def iob_to_label(label):
    label = label[2:]
    if not label:
        return 'other'
    return label

def scale_bounding_box(box, width_scale : float = 1.0, height_scale : float = 1.0) -> List[int]:
    return [
        int(box['left']),
        int(box['top']),
        int(box['width']),
        int(box['height'])
    ]
# run inference
def run_inference(path, model=model, processor=processor, output_image=True):
    # create model input
    # image = path.convert("RGB")
    # image = Image.open(path).convert("RGB")
    # width, height = image.size
    # width_scale = 1000 / width
    # height_scale = 1000 / height


    # # encoding = processor(image, return_tensors="pt")
    
    # # Perform OCR to extract text and bounding boxes
    
    # ocr_result = perform_ocr(image)  # Replace with your OCR implementation
    # words = []
    # boxes = []

    # for row in ocr_result:
    #     boxes.append(scale_bounding_box(row["bounding_box"],width_scale,height_scale))
    #     words.append(row["word"])
    # label_mapping = model.config.label2id
    original_labels = ['S-VALUE', 'I-VALUE', 'E-IGNORE', 'S-KEY', 'E-COLUMN', 'B-IGNORE', 'S-COLUMN', 'I-COLUMN', 'O', 'E-VALUE', 'B-COLUMN', 'S-IGNORE', 'I-IGNORE', 'B-VALUE']
    # Map original labels to model's label IDs
    dataset = load_from_disk('./raw_data_finetune_layoutml3')
    example = dataset["test"][0]
    image = example["image"].convert('RGB')
    words = example["tokens"]
    boxes = example["bboxes"]
    word_labels = example["ner_tags"]

    encoding = processor(words,image,boxes=boxes,word_labels=word_labels,return_tensors="pt")
    
    with torch.no_grad():
      outputs = model(**encoding)
    print('097887')
    labels = ['S-VALUE', 'I-VALUE', 'E-IGNORE', 'S-KEY', 'E-COLUMN', 'B-IGNORE', 'S-COLUMN', 'I-COLUMN', 'O', 'E-VALUE', 'B-COLUMN', 'S-IGNORE', 'I-IGNORE', 'B-VALUE']
    id2label = {v: k for v, k in enumerate(labels)}
    predictions = outputs.logits.argmax(-1).squeeze().tolist() 
    print(predictions)
    true_predictions = [id2label[prediction] for prediction in predictions]
    # token_boxes = encoding.bbox.squeeze().tolist()
    # true_boxes = [unnormalize_box(box, width, height) for box in token_boxes] 
    print(true_predictions) 
    # print(true_predictions)
    # key_value_pairs = [];
    # for prediction, box in zip(true_predictions, true_boxes):
        
    #     predicted_label = iob_to_label(prediction).lower()

    #     if([0.0, 0.0, 0.0, 0.0] != box):
    #         print(predicted_label)
    #         # cropped_image = image.crop(box)
    #         # extracted_text = pytesseract.image_to_string(cropped_image)
    #         # key_value_pairs.append({predicted_label: extracted_text})
    #         # print(extracted_text,predicted_label)
            
    #         # if(extracted_text != ''):
    #         #     print(prediction)
    #         #     print(extracted_text)


    # print(key_value_pairs)
    # get labels
    # print(model.config.id2label)
    labels = [model.config.id2label[prediction] for prediction in predictions]
    print(model.config.id2label)
    # print(labels)
    if output_image:
        return draw_boxes(image, encoding["bbox"][0], labels)
    else:
        return labels


result_image = run_inference('./lyoutml-ml-imgage for fine-tune/TS SP3 MND MOTOR 5041_June2023 + Annexures_page-0024.jpg')
result_image.save('output_image.jpeg')  