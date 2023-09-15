from transformers import LayoutLMForTokenClassification, LayoutLMv2Processor,LayoutLMv3ForTokenClassification,LayoutLMv3Processor,AutoProcessor,AutoModelForTokenClassification
from PIL import Image, ImageDraw, ImageFont
import torch
from pytesseract import Output, pytesseract
from typing import List
from datasets import load_dataset, load_from_disk
import random
import cv2
import numpy as np
from typing import List

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

# tuplify
def tup(point):
    return (point[0], point[1]);

# returns true if the two boxes overlap
def overlap(source, target):
    # unpack points
    tl1, br1 = source;
    tl2, br2 = target;

    # checks
    if (tl1[0] >= br2[0] or tl2[0] >= br1[0]):
        return False;
    if (tl1[1] >= br2[1] or tl2[1] >= br1[1]):
        return False;
    return True;

# returns all overlapping boxes
def getAllOverlaps(boxes, bounds, index):
    overlaps = [];
    for a in range(len(boxes)):
        if a != index:
            if overlap(bounds, boxes[a]):
                overlaps.append(a);
    return overlaps;

def medianCanny(img, thresh1, thresh2):
    median = np.median(img)
    img = cv2.Canny(img, int(thresh1 * median), int(thresh2 * median))
    return img

def extractBoundingBox(path):
  img = cv2.imread(path)
  orig = np.copy(img);
  blue, green, red = cv2.split(img)
  
  blue_edges = medianCanny(blue, 0, 1)
  green_edges = medianCanny(green, 0, 1)
  red_edges = medianCanny(red, 0, 1)
  
  edges = blue_edges | green_edges | red_edges
  
  # I'm using OpenCV 3.4. This returns (contours, hierarchy) in OpenCV 2 and 4
  contours,hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP ,cv2.CHAIN_APPROX_TC89_L1)
  # go through the contours and save the box edges
  boxes = []; # each element is [[top-left], [bottom-right]];
  hierarchy = hierarchy[0]
  for component in zip(contours, hierarchy):
      currentContour = component[0]
      currentHierarchy = component[1]
      x,y,w,h = cv2.boundingRect(currentContour)
      if currentHierarchy[3] < 0:
         new = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
         boxes.append([[x,y], [x+w, y+h]]);
  
  # filter out excessively large boxes
  filtered = [];
  max_area = 30000;
  for box in boxes:
      w = box[1][0] - box[0][0];
      h = box[1][1] - box[0][1];
      if w*h < max_area:
          filtered.append(box);
  boxes = filtered;

  # go through the boxes and start merging
  merge_margin = 20;
  
  # this is gonna take a long time
  finished = False;
  highlight = [[0,0], [1,1]];
  points = [[[0,0]]];
  while not finished:
      # set end con
      finished = True;
  
      # check progress
  
  
      # draw boxes # comment this section out to run faster
      copy = np.copy(orig);
      for box in boxes:
          cv2.rectangle(copy, tup(box[0]), tup(box[1]), (0,200,0), 1);
      cv2.rectangle(copy, tup(highlight[0]), tup(highlight[1]), (0,0,255), 2);
      for point in points:
          point = point[0];
          cv2.circle(copy, tup(point), 4, (255,0,0), -1);
      # cv2.imshow("Copy", copy);
      key = cv2.waitKey(1);
      if key == ord('q'):
          break;
  
      # loop through boxes
      index = 0;
      while index < len(boxes):
          # grab current box
          curr = boxes[index];
  
          # add margin
          tl = curr[0][:];
          br = curr[1][:];
          tl[0] -= merge_margin;
          tl[1] -= merge_margin;
          br[0] += merge_margin;
          br[1] += merge_margin;
  
          # get matching boxes
          overlaps = getAllOverlaps(boxes, [tl, br], index);
  
          # check if empty
          if len(overlaps) > 0:
              # combine boxes
              # convert to a contour
              con = [];
              overlaps.append(index);
              for ind in overlaps:
                  tl, br = boxes[ind];
                  con.append([tl]);
                  con.append([br]);
              con = np.array(con);
  
              # get bounding rect
              x,y,w,h = cv2.boundingRect(con);
  
              # stop growing
              w -= 1;
              h -= 1;
              merged = [[x,y], [x+w, y+h]];
  
              # highlights
              highlight = merged[:];
              points = con;
  
              # remove boxes from list
              overlaps.sort(reverse = True);
              for ind in overlaps:
                  del boxes[ind];
              boxes.append(merged);
  
              # set flag
              finished = False;
              break;
  
          # increment
          index += 1;
  cv2.destroyAllWindows();

  # show final
  # copy = np.copy(orig);
  # for box in boxes:
  #     cv2.rectangle(copy, tup(box[0]), tup(box[1]), (0,200,0), 1);
  # cv2_imshow(copy)
  # cv2.waitKey(0)
  main_image = Image.open(path).convert("RGB")
  ocr_data = []

  for box in boxes:
    cv2.rectangle(copy, tup(box[0]), tup(box[1]), (0,200,0), 1);


  for box in boxes:
    p1,p2 = box
    bounding_boxes = {
    "left": p1[0],
    "top": p1[1], 
    "right": p2[0],
    "bottom": p2[1]
    }
    
    cropped_image = main_image.crop([p1[0]-5,p1[1]-5,p2[0]+5,p2[1]+5])

    image_array = np.array(cropped_image)
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    extracted_text = pytesseract.image_to_string(gray,config='— oem 3 — psm 10',lang='eng')
    

    # print('2')
    # print(extracted_text_2)
    if not extracted_text.strip():
      extracted_text = pytesseract.image_to_string(cropped_image,config='--psm 10')
      
    # cv2_imshow(gray)
    print(extracted_text)
    ocr_data.append({"word": extracted_text, "bounding_box": bounding_boxes})
    # # cv2_imshow(image_array)
    # print(extracted_text)
    # print('----')
  return ocr_data


def run_inference(path, model=model, processor=processor, output_image=True):
    
    image = Image.open(path).convert("RGB")
    width, height = image.size
    width_scale = 1000 / width
    height_scale = 1000 / height
    ocr_result = extractBoundingBox(path)  # Replace with your OCR implementation
    
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
            # custom_config = '--psm 10'
            # extracted_text = pytesseract.image_to_string(cropped_image,config=custom_config)
            image_array = np.array(cropped_image)
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            extracted_text = pytesseract.image_to_string(gray,config='— oem 3 — psm 10',lang='eng')
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
    

    
result_image = run_inference('./lyoutml-ml-imgage for fine-tune/TS SP3 MND MOTOR 5041_June2023 + Annexures_page-0073.jpg')
result_image.save('output_image.jpeg')  