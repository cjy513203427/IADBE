from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO('/home/jinyao/PycharmProjects/IADBE/yolo/yolov8_project/train_experiment/weights/best.pt')

# Perform prediction on the specified image
results = model.predict(source='/home/jinyao/PycharmProjects/IADBE/datasets/Custom_Dataset/anomaly-detection.v2i.yolov8/valid/images/20230522124919_png.rf.2dcb75d340f86c8b412d55ce20bd3bc2.jpg')

# Iterate over the results and display or save them
for result in results:
    result.show()  # Display the image with predictions
    result.save(filename='inferenced_image.jpg')  # Save the results to an output directory
