#set environment variable in lambda 
# (lambda function->configuration->environment variable->give key and value(key as "ENDPOINT_NAME" value as sagemaker's endpoint name))



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
    #    img = image.read()
        #print(type(img))
    ##img = Image.fromarray(img)
    success, encoded_image = cv2.imencode('.png', img)
    img = encoded_image.tobytes()
    print(type(img))
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
    return 0.7

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
    if(orb_similarity > 0.79):
      return "IDLE"
    else:
      return "RUNNING"

def lambda_handler(event, context):
    # TODO implement
    s3 = boto3.client("s3")
    s3.download_file('opencvtutorial', 'input_feed_5fps.mp4', '/tmp/input_feed.mp4')
    input_feed = "/tmp/input_feed.mp4"
    
    #input video path
    cap = cv2.VideoCapture(input_feed)
    ret, current_frame = cap.read()
    print("Invoking Model")
    normalized_boxes, classes_names, confidences = model_prediction(current_frame)
    print("Model Prediction Successful")
    bbox = get_bbox(current_frame,normalized_boxes)
    current_frame_cropped = crop_image(current_frame,bbox)
    previous_frame_cropped = current_frame_cropped
    video_frame = current_frame.copy()
    height, width, layers = video_frame.shape
    size = (width,height)
    out = cv2.VideoWriter('/tmp/inference_feed.avi',cv2.VideoWriter_fourcc(*'XVID'), 5, size)
    last_state = []
    frame = 0
    logs = {}
    while(cap.isOpened() & ret == True):
        frame+=1
        state = []
        for i in range(len(current_frame_cropped)):
            state.append(get_state(current_frame_cropped[i], previous_frame_cropped[i]))
        last_state = state
        temp = {}
        video_frame = current_frame.copy()
        for i in range(len(bbox)):
            color = ()
            if state[i] == "RUNNING":
                color = (0,255,0)
                temp["windmill"+str(i)] = "RUNNING"
            else: 
                color = (0,0,255)
                temp["windmill"+str(i)] = "IDLE"
            x,y,w,h = bbox[i]
            cv2.rectangle(video_frame,(x,y),(x+w,y+h),color,2)
            cv2.putText(video_frame,state[i],(x,y-3),0,1,color,2)
        logs["frame"+str(frame)+"_time "+str(frame/5)+'s'] = temp
        out.write(video_frame)
    
        for i in range(len(current_frame_cropped)):
            previous_frame_cropped[i] = current_frame_cropped[i].copy()
    
        ret , current_frame = cap.read()
        if(ret == True):
            current_frame_cropped = crop_image(current_frame,bbox)

    print("Uploading Inference Video......") 
    s3.put_object(Bucket="opencvtutorial", Key="inference_feed.avi", Body=open("/tmp/inference_feed.avi","rb").read())
    
    result = {}
    for i in range(len(last_state)):
        if(last_state[i] == "IDLE"):
            result["Equipment0"+str(i+1)] = str(10001085+i)
    if(len(result)==0):
        for i in range(len(last_state)):
            result["Equipment0"+str(i+1)] = str(10001085+i)
        result["FunctLoc"] = "MUM1-THA-AB-02"
        result["ShortText"] = "Fan Blades are Moving"
        result["LongText"] = "WIndmill Fan blades are in RUNNING state"
    else:
        result["FunctLoc"] = "MUM1-THA-AB-02"
        result["ShortText"] = "Fan Blades are Not Moving"
        result["LongText"] = "WIndmill Fan blades are in IDLE state not moving"
        
    return {"d" : result}
        
    '''return {
        'logs' : logs,
        'last_state' : last_state,
        'result' : result,
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!'),
    }'''
