import json
import os 
import io 
import boto3 
from io import BytesIO
from PIL import Image
import numpy as np
import base64
import cv2

# grab environment variables 

ENDPOINT_NAME = os.environ['ENDPOINT_NAME'] 
runtime= boto3.client('runtime.sagemaker')

def parse_response(query_response):
    print("Hello Parse")
    model_predictions = json.loads(query_response)
    normalized_boxes, classes, scores, labels = (
        model_predictions["normalized_boxes"],
        model_predictions["classes"],
        model_predictions["scores"],
        model_predictions["labels"],
    )
    # Substitute the classes index with the classes name
    class_names = [labels[int(idx)] for idx in classes]
    print("Bye Bye Parse")
    return normalized_boxes, class_names, scores

def model_prediction(img):
    s3 = boto3.client('s3')
    #s3.download_file('opencvtutorial', 'test_image.png', '/tmp/test_image.png')
    #input_file = "/tmp/test_image.png"
    #with open(input_file, "rb") as image:
    #    b = image.read()
    img = Image.fromarray(img)
    try:
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/x-image',
            Accept= "application/json;verbose;n_predictions=3",
            Body=img
        )
        query_response = response['Body'].read().decode('utf-8')
        #print(query_response)
        print("Hello world!")
        normalized_boxes, classes_names, confidences = parse_response(query_response)
        return (normalized_boxes, classes_names, confidences)
    except Exception as e:
        print("Inference Error:")
        print(e)

def get_bbox(img, normalized_boxes):
    image_np = np.array(img)
    bbox = []
    for idx in range(len(normalized_boxes)):
        left, bot, right, top = normalized_boxes[idx]
        x, w = [val * image_np.shape[1] for val in [left, right - left]]
        y, h = [val * image_np.shape[0] for val in [bot, top - bot]]
        bbox.append([int(x),int(y),int(w),int(h)])
    return bbox   

def crop_image(img,bbox):
    cropped_images = []
    for i in range(len(bbox)):
        X,Y,W,H = bbox[i]
        cropped_image = img[Y:Y+H, X:X+W]
        cropped_images.append(cropped_image.copy())
    return cropped_images

def orb_sim(img1, img2):
  
  orb = cv2.ORB_create()

  # detect keypoints and descriptors
  kp_a, desc_a = orb.detectAndCompute(img1, None)
  kp_b, desc_b = orb.detectAndCompute(img2, None)

  # define the bruteforce matcher object
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
  #perform matches.
  #print(type(desc_a))
  #print(type(desc_b))
  try:
    matches = bf.match(desc_a, desc_b)
  except:
    return 0.97

  #Look for similar regions with distance < 50. Goes from 0 to 100 so pick a number between.
  #lesser the number more the sensitivity
  similar_regions = [i for i in matches if i.distance < 5]  
  if len(matches) == 0:
    return 0
  return len(similar_regions) / len(matches)

def get_state(current_frame,previous_frame):
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

    orb_similarity = orb_sim(current_frame_gray, previous_frame_gray)
    if(orb_similarity > 0.98):
      return "IDLE"
    else:
      return "RUNNING"

def lambda_handler(event, context):
    # TODO implement
    s3 = boto3.client("s3")
    s3.download_file('opencvtutorial', 'input_feed.mp4', '/tmp/input_feed.mp4')
    input_feed = "/tmp/input_feed.mp4"
    
    #input video path
    cap = cv2.VideoCapture(input_feed)
    ret, current_frame = cap.read()
    print("Invoking Model")
    print(model_prediction(current_frame))
    normalized_boxes, classes_names, confidences = model_prediction(current_frame)
    '''normalized_boxes, classes_names, confidences = [
  [
    [
      0.4243313289423871,
      0.13580453395843506,
      0.6603998528703194,
      0.6410654187202454
    ],
    [
      0.6078712362549904,
      0.32311689853668213,
      0.7160287596580741,
      0.7545114755630493
    ],
    [
      0.05837434609030837,
      0.29826390743255615,
      0.18896575121102355,
      0.6512647271156311
    ]
  ],
  [
    "windturbine",
    "windturbine",
    "windturbine"
  ],
  [
    0.6736352443695068,
    0.7843161821365356,
    0.7976691722869873
  ]
]'''
    print("Model Prediction Successful")
    bbox = get_bbox(current_frame,normalized_boxes)
    current_frame_cropped = crop_image(current_frame,bbox)
    previous_frame_cropped = current_frame_cropped
    video_frame = current_frame.copy()
    height, width, layers = video_frame.shape
    size = (width,height)
    out = cv2.VideoWriter('/tmp/inference_feed.avi',cv2.VideoWriter_fourcc(*'XVID'), 30, size)
    last_state = []
    while(cap.isOpened() & ret == True):
        state = []
        for i in range(len(current_frame_cropped)):
            state.append(get_state(current_frame_cropped[i], previous_frame_cropped[i]))
        
        video_frame = current_frame.copy()
        for i in range(len(bbox)):
            color = ()
            if state[i] == "RUNNING":
                color = (0,255,0)
            else: color = (0,0,255)
            x,y,w,h = bbox[i]
            cv2.rectangle(video_frame,(x,y),(x+w,y+h),color,2)
            cv2.putText(video_frame,state[i],(x,y-3),0,1,color,2)
            last_state = state
        out.write(video_frame)
    
        for i in range(len(current_frame_cropped)):
            previous_frame_cropped[i] = current_frame_cropped[i].copy()
    
        ret , current_frame = cap.read()
        if(ret == True):
            current_frame_cropped = crop_image(current_frame,bbox)

    print("Uploading Inference Video......") 
    s3.put_object(Bucket="opencvtutorial", Key="inference_feed.avi", Body=open("/tmp/inference_feed.avi","rb").read())
    
    #return {"img": "ok"}
    return {
        'state' : {
                                        "Equipment01": "10001085",
                                        "Equipment02": "10001086",
                                        "Equipment03": "10001087",
                                        "FunctLoc": "MUM1-THA-AB-02",
                                        "ShortText": "Fan Blades Not Moving",
                                        "LongText":"WIndmill Fan blades in IDLE state not moving"
                                    },
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
