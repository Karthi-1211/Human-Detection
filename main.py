import cv2
import imutils
import numpy as np
import argparse
import os

def detect(frame, min_confidence=0.5):
    # Convert frame to grayscale (HOG works on grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect people with HOG
    bounding_box_coordinates, weights = HOGCV.detectMultiScale(
        gray, 
        winStride=(4, 4), 
        padding=(16, 16),  # Increased padding for better detection
        scale=1.05,        # Smaller scale step for finer detection
        useMeanshiftGrouping=False
    )
    
    person = 1
    detected_boxes = []
    
    # Filter detections based on confidence
    for i, (x, y, w, h) in enumerate(bounding_box_coordinates):
        if weights[i] >= min_confidence:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'Person {person} ({weights[i]:.2f})', (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            detected_boxes.append((x, y, w, h))
            person += 1
    
    # Display status and count
    cv2.putText(frame, 'Status: Detecting', (40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f'Total Persons: {person - 1}', (40, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.imshow('Output', frame)

    return frame, person - 1, detected_boxes

def detectByPathVideo(path, writer):
    video = cv2.VideoCapture(path)
    check, frame = video.read()
    if not check:
        print(f'Error: Video not found at {path}. Please provide a valid path.')
        return

    print('Detecting people in video...')
    frame_count = 0
    while video.isOpened():
        check, frame = video.read()
        if not check:
            break

        frame_count += 1
        frame = imutils.resize(frame, width=min(800, frame.shape[1]))
        frame, persons_detected, boxes = detect(frame)
        
        if writer is not None:
            writer.write(frame)
        
        print(f'Frame {frame_count}: Detected {persons_detected} persons')
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    
    video.release()
    cv2.destroyAllWindows()

def detectByCamera(writer):
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print('Error: Could not open webcam.')
        return

    print('Detecting people via webcam...')
    frame_count = 0
    while True:
        check, frame = video.read()
        if not check:
            print('Error: Failed to capture frame from webcam.')
            break

        frame_count += 1
        frame = imutils.resize(frame, width=min(800, frame.shape[1]))
        frame, persons_detected, boxes = detect(frame)
        
        if writer is not None:
            writer.write(frame)
        
        print(f'Frame {frame_count}: Detected {persons_detected} persons')
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    
    video.release()
    cv2.destroyAllWindows()

def detectByPathImage(path, output_path):
    image = cv2.imread(path)
    if image is None:
        print(f'Error: Unable to read image at {path}. Check if the file exists.')
        return

    print(f'Processing image: {path}')
    image = imutils.resize(image, width=min(800, image.shape[1]))
    result_image, persons_detected, boxes = detect(image)
    
    print(f'Total Persons Detected: {persons_detected}')
    for i, (x, y, w, h) in enumerate(boxes):
        print(f'Person {i + 1}: (x={x}, y={y}, w={w}, h={h})')

    cv2.imshow('Detected Image', result_image)
    if output_path is not None:
        cv2.imwrite(output_path, result_image)
        print(f'Output saved to: {output_path}')

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def humanDetector(args):
    image_path = args["image"]
    video_path = args["video"]
    camera = args["camera"].lower() == 'true'

    writer = None
    if args['output'] is not None and image_path is None:
        writer = cv2.VideoWriter(args['output'], cv2.VideoWriter_fourcc(*'MJPG'), 10, (800, 600))

    if camera:
        print('[INFO] Opening Webcam.')
        detectByCamera(writer)
    elif video_path is not None:
        print('[INFO] Opening Video from path.')
        detectByPathVideo(video_path, writer)
    elif image_path is not None:
        print('[INFO] Opening Image from path.')
        detectByPathImage(image_path, args['output'])
    else:
        print('Error: No input provided. Use --image, --video, or --camera.')

def argsParser():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-v", "--video", default=None, help="Path to video file")
    arg_parse.add_argument("-i", "--image", default=None, help="Path to image file")
    arg_parse.add_argument("-c", "--camera", default="False", help="Set to 'True' to use webcam")
    arg_parse.add_argument("-o", "--output", type=str, default=None, help="Path to optional output video/image file")
    args = vars(arg_parse.parse_args())
    return args

if __name__ == "__main__":
    print("Initializing HOG Descriptor...")
    HOGCV = cv2.HOGDescriptor()
    HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    print("HOG Descriptor initialized.")

    args = argsParser()
    print("Arguments:", args)
    humanDetector(args)