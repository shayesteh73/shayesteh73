from imageai.Detection import ObjectDetection
import os


model_path = r"I:\model"
images_path = r"H:\axe"

# Checking the existence of paths.
if not os.path.exists(model_path):
    print(f"The model path is invalid: {model_path}")
    exit()
if not os.path.exists(images_path):
    print(f"The image path is invalid: {images_path}")
    exit()

# Checking the existence of the model file
model_file = os.path.join(model_path, "yolov3.pt")
if not os.path.isfile(model_file):
    print(f"Model file not found: {model_file}")
    exit()

# Check the existence of input image files
input_image_file = os.path.join(images_path, "DKaC6a7W0AAyCnv.jpg")
if not os.path.isfile(input_image_file):
    print(f"Image input file not found: {input_image_file}")
    exit()

try:
    # Setting up and loading the model
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(model_file)  # model file path
    detector.loadModel()
    
    # Object detection in the image
    output_image_file = os.path.join(images_path, "DKaC6a7W0AAyCnv_detected.jpg")
    detection = detector.detectObjectsFromImage(
        input_image=input_image_file,
        output_image_path=output_image_file  # Using a different name for the output image
    )

    # Printing detection results
    for eachObject in detection:
        print(eachObject["name"], " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"])
        print("--------------------------------")

except Exception as e:
    print(f"یک خطای زمان اجرا رخ داد: {str(e)}")



detection


custom_detector = detector.CustomObjects(bus=True)
custom_detection = detector.detectObjectsFromImage(custom_objects=custom_detector,input_image=os.path.join(images_path, "DKaC6a7W0AAyCnv.jpg"), output_image_path = os.path.join(images_path, "DKaC6a7W0AAyCnv_customdetected.jpg"))

custom_detection                           



custom_detector = detector.CustomObjects(bus=True)
custom_detection = detector.detectObjectsFromImage(custom_objects=custom_detector,input_image=os.path.join(images_path, "DKaC6a7W0AAyCnv.jpg"), output_image_path = os.path.join(images_path, "DKaC6a7W0AAyCnv_customdetected.jpg"),minimum_percentage_probability=80)

custom_detection                           

try:
    custom_detector = detector.CustomObjects(bus=True)
    custom_detection = detector.detectObjectsFromImage(
        custom_objects=custom_detector,
        input_image=input_image_file,
        output_image_path=os.path.join(images_path, "DKaC6a7W0AAyCnv_customdetected.jpg"),
        minimum_percentage_probability=80
    )

    # Printing custom detection results
    for eachObject in custom_detection:
        print(eachObject["name"], " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"])
        print("--------------------------------")

except Exception as e:
    print(f"یک خطای زمان اجرا رخ داد: {str(e)}")
