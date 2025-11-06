# #matplotlib inline
# import os, glob

# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # Disable oneDNN logs

# import tensorflow as tf

# import numpy as np

# import matplotlib.pyplot as plt
# import cv2
# from PIL import Image
# import warnings
# warnings.filterwarnings('ignore')
# import tensorflow as tf 
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
# data_path = r'C:\Users\GURPREET\Desktop\traffic_sign_detection_gtsrb\archive\\'
# train_path = data_path + 'Train/'
# test_path = data_path + 'Test/'
# df_meta = pd.read_csv(data_path + 'Meta.csv')
# df_train = pd.read_csv(data_path + 'Train.csv')
# df_test = pd.read_csv(data_path + 'Test.csv')
# df_train.head()
# import argparse

# # inference_vscode.py
# import os
# import numpy as np
# import cv2
# import tensorflow as tf
# # Load metadata
# import csv
# print(tf.__file__)
# print(dir(tf))
# from tensorflow.keras.models import load_model

# # Suppress TensorFlow warnings & oneDNN logs
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# # -----------------------------
# # 1. Load trained model
# # -----------------------------
# MODEL_PATH = r"C:\Users\GURPREET\Desktop\damn\traffic_sign_detection_gtsrb\models\traffic_sign_detection_gtsrb.h5"   # Path to your saved model
# model = load_model(MODEL_PATH)
# print("[INFO] Model loaded successfully!")
# print("[DEBUG] Model output layer units:", model.output_shape)
# # Define class names (0-42 for GTSRB)
# # -----------------------------
# # 1.1 Define class names (14 classes)
# # -----------------------------


# # data_path = r"C:\Users\GURPREET\Desktop\traffic_sign_detection_gtsrb\archive\\"

# # with open(data_path + "Meta.csv", newline='', encoding='utf-8') as csvfile:
# #     reader = csv.DictReader(csvfile)
# #     print("[DEBUG] CSV fieldnames:", reader.fieldnames)  # 👈 print headers


# # data_path = r"C:\Users\GURPREET\Desktop\traffic_sign_detection_gtsrb\archive\\"
# # id_to_sign = {}

# # with open(data_path + "Meta.csv", newline='', encoding='utf-8') as csvfile:
# #     reader = csv.DictReader(csvfile)
# #     for row in reader:
# #         id_to_sign[int(row["ClassId"])] = row["SignId"]
# # class_names = [id_to_sign[i] for i in range(len(id_to_sign))]
# # print(f"[INFO] Loaded {len(class_names)} class names")

# # Load class names from Meta.csv
# # data_path = r"C:\Users\GURPREET\Desktop\traffic_sign_detection_gtsrb\archive\\"
# # class_names = []

# # with open(data_path + "Meta.csv", newline='', encoding='utf-8') as csvfile:
# #     reader = csv.DictReader(csvfile)
# #     rows = sorted(reader, key=lambda row: int(row["ClassId"]))
# #     class_names = [row["SignID"] for row in rows]

# # print(f"[INFO] Loaded {len(class_names)} class names")




# class_names = [
#     "Speed limit (20km/h)",        # 0
#     "Speed limit (30km/h)",        # 1
#     "Speed limit (50km/h)",        # 2
#     "Speed limit (60km/h)",        # 3
#     "Speed limit (70km/h)",        # 4
#     "Speed limit (80km/h)",        # 5
#     "End of speed limit (80km/h)", # 6
#     "Speed limit (100km/h)",       # 7
#     "Speed limit (120km/h)",       # 8
#     "No passing",                  # 9
#     "No passing for vehicles over 3.5 metric tons", # 10
#     "Right-of-way at the next intersection",        # 11
#     "Priority road",               # 12
#     "Yield",                       # 13
#     "Stop",                        # 14
#     "No vehicles",                 # 15
#     "Vehicles over 3.5 metric tons prohibited",     # 16
#     "No entry",                    # 17
#     "General caution",              # 18
#     "Dangerous curve to the left", # 19
#     "Dangerous curve to the right",# 20
#     "Double curve",                # 21
#     "Bumpy road",                  # 22
#     "Slippery road",               # 23
#     "Road narrows on the right",   # 24
#     "Road work",                   # 25
#     "Traffic signals",             # 26
#     "Pedestrians",                 # 27
#     "Children crossing",           # 28
#     "Bicycles crossing",           # 29
#     "Beware of ice/snow",          # 30
#     "Wild animals crossing",       # 31
#     "End of all speed and passing limits", # 32
#     "Turn right ahead",            # 33
#     "Turn left ahead",             # 34
#     "Ahead only",                  # 35
#     "Go straight or right",        # 36
#     "Go straight or left",         # 37
#     "Keep right",                  # 38
#     "Keep left",                   # 39
#     "Roundabout mandatory",        # 40
#     "End of no passing",           # 41
#     "End of no passing by vehicles over 3.5 metric tons" # 42
# ]

# # -----------------------------
# # 2. Define helper function
# # -----------------------------
# def preprocess_image(img_path, target_size=(32, 32)):
#     """
#     Preprocess image for prediction
#     - Reads image
#     - Resizes
#     - Normalizes
#     """
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR -> RGB
#     img = cv2.resize(img, target_size)
#     img = img.astype("float32") / 255.0
#     img = np.expand_dims(img, axis=0)  # Add batch dimension
#     return img

# def predict_image(img_path):
#     """
#     Predict traffic sign class for a single image
#     """
#     img = preprocess_image(img_path)
#     preds = model.predict(img)
#     class_id = np.argmax(preds, axis=1)[0]
#     confidence = np.max(preds)
#     return class_id, confidence

# # -----------------------------
# # 3. Test the script
# # -----------------------------
# if __name__ == "__main__":
#     # Example test image (replace with your actual test path)
#     test_image_path = "D:\traffic_sign_detection_gtsrb\data\Test\00009.png"   # Stop sign
#     class_id, confidence = predict_image(test_image_path)

#     print(f"[RESULT] Predicted Class ID: {class_id}, Confidence: {confidence:.2f}")
#     print(f"[RESULT] Predicted Class ID: {class_id}, "
#           f"Label: {class_names[class_id]}, "
#           f"Confidence: {confidence:.2f}")








# # -----------------------------
# # Traffic Sign Detection - Inference Script
# # -----------------------------

# import os
# import glob
# import numpy as np
# import cv2
# import tensorflow as tf
# from tensorflow.keras.models import load_model

# # -----------------------------
# # 1. Suppress TensorFlow warnings
# # -----------------------------
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# # -----------------------------
# # 2. Load trained model
# # -----------------------------
# MODEL_PATH = r"C:\Users\GURPREET\Desktop\damn\traffic_sign_detection_gtsrb\models\traffic_sign_detection_gtsrb.h5"
# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# model = load_model(MODEL_PATH)
# print("[INFO] Model loaded successfully!")

# # -----------------------------
# # 3. Define 43 GTSRB class names
# # -----------------------------
# class_names = [
#     "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)", "Speed limit (60km/h)",
#     "Speed limit (70km/h)", "Speed limit (80km/h)", "End of speed limit (80km/h)", "Speed limit (100km/h)",
#     "Speed limit (120km/h)", "No passing", "No passing for vehicles over 3.5 metric tons",
#     "Right-of-way at the next intersection", "Priority road", "Yield", "Stop", "No vehicles",
#     "Vehicles over 3.5 metric tons prohibited", "No entry", "General caution", "Dangerous curve to the left",
#     "Dangerous curve to the right", "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
#     "Road work", "Traffic signals", "Pedestrians", "Children crossing", "Bicycles crossing", "Beware of ice/snow",
#     "Wild animals crossing", "End of all speed and passing limits", "Turn right ahead", "Turn left ahead",
#     "Ahead only", "Go straight or right", "Go straight or left", "Keep right", "Keep left", "Roundabout mandatory",
#     "End of no passing", "End of no passing by vehicles over 3.5 metric tons"
# ]

# # -----------------------------
# # 4. Helper functions
# # -----------------------------
# def preprocess_image(img_path, target_size=(32, 32)):
#     """
#     Reads, resizes, normalizes, and adds batch dimension to an image
#     """
#     print(f"[DEBUG] Loading image: {img_path}")
#     img = cv2.imread(img_path)
#     if img is None:
#         raise FileNotFoundError(f"[ERROR] Could not load image at {img_path}. "
#                                 "Check file path and extension.")
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, target_size)
#     img = img.astype("float32") / 255.0
#     img = np.expand_dims(img, axis=0)
#     return img

# def predict_image(img_path):
#     """
#     Predicts class ID and confidence for a single image
#     """
#     img = preprocess_image(img_path)
#     preds = model.predict(img)
#     class_id = int(np.argmax(preds, axis=1)[0])
#     confidence = float(np.max(preds))
#     return class_id, confidence

# # -----------------------------
# # 5. Select a test image automatically
# # -----------------------------
# TEST_FOLDER = r"C:\Users\GURPREET\Desktop\traffic_sign_detection_gtsrb\archive\Test\*"
# test_images = glob.glob(TEST_FOLDER)
# if len(test_images) == 0:
#     raise FileNotFoundError(f"No test images found in folder: {TEST_FOLDER}")

# test_image_path = test_images[0]  # pick the first image
# print(f"[INFO] Using test image: {test_image_path}")

# # -----------------------------
# # 6. Run prediction
# # -----------------------------
# class_id, confidence = predict_image(test_image_path)
# print(f"[RESULT] Predicted Class ID: {class_id}, "
#       f"Label: {class_names[class_id]}, "
#       f"Confidence: {confidence*100:.2f}%")

# # -----------------------------
# # 7. Display the image with prediction
# # -----------------------------
# img = cv2.imread(test_image_path)
# if img is None:
#     raise FileNotFoundError(f"Could not load image for display: {test_image_path}")

# # Prepare label text with class name and confidence %
# label_text = f"{class_names[class_id]} ({confidence*100:.2f}%)"

# # Draw label on the image
# cv2.putText(img, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
#             0.8, (0, 255, 0), 2)

# # Optional: resize image for better display
# img_display = cv2.resize(img, (1000, 1000))

# # Show image in a window
# cv2.imshow("Prediction", img_display)
# cv2.waitKey(0)
# cv2.destroyAllWindows()









































































# # -----------------------------
# # Traffic Sign Detection - Inference
# # -----------------------------

# import os
# import glob
# import numpy as np
# import cv2
# import tensorflow as tf
# from tensorflow.keras.models import load_model

# # -----------------------------
# # 1. Suppress TensorFlow warnings
# # -----------------------------
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# # -----------------------------
# # 2. Load trained model
# # -----------------------------
# MODEL_PATH = r"C:\Users\GURPREET\Desktop\damn\traffic_sign_detection_gtsrb\models\traffic_sign_detection_gtsrb.h5"
# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# model = load_model(MODEL_PATH)
# print("[INFO] Model loaded successfully!")

# # -----------------------------
# # 3. Define 43 GTSRB class names
# # -----------------------------
# class_names = [
#     "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)", "Speed limit (60km/h)",
#     "Speed limit (70km/h)", "Speed limit (80km/h)", "End of speed limit (80km/h)", "Speed limit (100km/h)",
#     "Speed limit (120km/h)", "No passing", "No passing for vehicles over 3.5 metric tons",
#     "Right-of-way at the next intersection", "Priority road", "Yield", "Stop", "No vehicles",
#     "Vehicles over 3.5 metric tons prohibited", "No entry", "General caution", "Dangerous curve to the left",
#     "Dangerous curve to the right", "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
#     "Road work", "Traffic signals", "Pedestrians", "Children crossing", "Bicycles crossing", "Beware of ice/snow",
#     "Wild animals crossing", "End of all speed and passing limits", "Turn right ahead", "Turn left ahead",
#     "Ahead only", "Go straight or right", "Go straight or left", "Keep right", "Keep left", "Roundabout mandatory",
#     "End of no passing", "End of no passing by vehicles over 3.5 metric tons"
# ]

# # -----------------------------
# # 4. Helper functions
# # -----------------------------
# def preprocess_image(img_path, target_size=(32, 32)):
#     """
#     Read, resize, normalize, and add batch dimension
#     """
#     print(f"[DEBUG] Loading image: {img_path}")
#     img = cv2.imread(img_path)
#     if img is None:
#         raise FileNotFoundError(f"[ERROR] Could not load image at {img_path}. "
#                                 "Check file path and extension.")
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, target_size)
#     img = img.astype("float32") / 255.0
#     img = np.expand_dims(img, axis=0)
#     return img

# def predict_image(img_path):
#     """
#     Predict class ID and confidence for a single image
#     """
#     img = preprocess_image(img_path)
#     preds = model.predict(img)
#     class_id = int(np.argmax(preds, axis=1)[0])
#     confidence = float(np.max(preds))
#     return class_id, confidence

# # -----------------------------
# # 5. Select a test image automatically
# # -----------------------------
# TEST_FOLDER = r"C:\Users\GURPREET\Desktop\traffic_sign_detection_gtsrb\archive\Test\*"
# test_images = glob.glob(TEST_FOLDER)
# if len(test_images) == 0:
#     raise FileNotFoundError(f"No test images found in folder: {TEST_FOLDER}")

# test_image_path = test_images[22]  # pick the first image
# print(f"[INFO] Using test image: {test_image_path}")

# # -----------------------------
# # 6. Run prediction
# # -----------------------------
# class_id, confidence = predict_image(test_image_path)
# print(f"[RESULT] Predicted Class ID: {class_id}, "
#       f"Label: {class_names[class_id]}, "
#       f"Confidence: {confidence*100:.2f}%")






#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! WRONG CODE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!# # -----------------------------
# # Traffic Sign Detection - Batch Inference and Evaluation
# import os
# import cv2
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import load_model
# from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# # -----------------------------
# # 0. Suppress TensorFlow warnings
# # -----------------------------
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# # -----------------------------
# # 1. Load trained model
# # -----------------------------
# MODEL_PATH = r"C:\Users\GURPREET\Desktop\damn\traffic_sign_detection_gtsrb\models\traffic_sign_detection_gtsrb.h5"
# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# model = load_model(MODEL_PATH)
# print("[INFO] Model loaded successfully!")

# # -----------------------------
# # 2. Load training CSV
# # -----------------------------
# TRAIN_CSV = r"C:\Users\GURPREET\Desktop\traffic_sign_detection_gtsrb\archive\Train.csv"
# df_train = pd.read_csv(TRAIN_CSV)
# print(f"[INFO] Total training images: {len(df_train)}")

# # -----------------------------
# # 3. Load or Fix Test CSV Automatically
# # -----------------------------
# TEST_CSV = r"C:\Users\GURPREET\Desktop\traffic_sign_detection_gtsrb\archive\Test.csv"
# TEST_DIR = r"C:\Users\GURPREET\Desktop\traffic_sign_detection_gtsrb\archive\Test"

# if not os.path.exists(TEST_CSV):
#     print("[WARNING] Test.csv not found! Generating automatically...")
#     data = []
#     subfolders = [f for f in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, f))]
#     if subfolders:
#         # Folder-based structure (Test/0/, Test/1/, ...)
#         for class_id in subfolders:
#             class_folder = os.path.join(TEST_DIR, class_id)
#             for img_name in os.listdir(class_folder):
#                 if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
#                     data.append([os.path.join(TEST_DIR, class_id, img_name), int(class_id)])
#         df_test = pd.DataFrame(data, columns=["Path", "ClassId"])
#     else:
#         # Flat structure (official GTSRB)
#         images = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#         df_test = pd.DataFrame({"Path": [os.path.join(TEST_DIR, img) for img in images]})
#         df_test["ClassId"] = 0  # placeholder
#     df_test.to_csv(TEST_CSV, index=False)
#     print(f"[INFO] Test.csv created successfully at: {TEST_CSV}")

# else:
#     df_test = pd.read_csv(TEST_CSV)
#     print("[INFO] Loaded existing Test.csv")

# # ------------------------------------------------
# # Fix inconsistent path formats automatically
# # ------------------------------------------------
# def fix_test_path(x):
#     if os.path.exists(x):
#         return x
#     filename = os.path.basename(x)
#     full_path = os.path.join(TEST_DIR, filename)
#     if os.path.exists(full_path):
#         return full_path
#     parts = x.replace("\\", "/").split("/")
#     if len(parts) >= 2 and parts[-2].isdigit():
#         class_id = parts[-2]
#         full_path = os.path.join(TEST_DIR, class_id, parts[-1])
#         if os.path.exists(full_path):
#             return full_path
#     return x

# df_test["Path"] = df_test["Path"].apply(fix_test_path)

# print("\n[DEBUG] Sample test paths after auto-fix:")
# print(df_test.sample(5))
# print(f"[INFO] Total test images detected: {len(df_test)}")

# # -----------------------------
# # 4. Define 43 GTSRB class names
# # -----------------------------
# class_names = [
#     "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)", "Speed limit (60km/h)",
#     "Speed limit (70km/h)", "Speed limit (80km/h)", "End of speed limit (80km/h)", "Speed limit (100km/h)",
#     "Speed limit (120km/h)", "No passing", "No passing for vehicles over 3.5 metric tons",
#     "Right-of-way at the next intersection", "Priority road", "Yield", "Stop", "No vehicles",
#     "Vehicles over 3.5 metric tons prohibited", "No entry", "General caution", "Dangerous curve to the left",
#     "Dangerous curve to the right", "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
#     "Road work", "Traffic signals", "Pedestrians", "Children crossing", "Bicycles crossing", "Beware of ice/snow",
#     "Wild animals crossing", "End of all speed and passing limits", "Turn right ahead", "Turn left ahead",
#     "Ahead only", "Go straight or right", "Go straight or left", "Keep right", "Keep left", "Roundabout mandatory",
#     "End of no passing", "End of no passing by vehicles over 3.5 metric tons"
# ]

# # -----------------------------
# # 5. Load and preprocess test images
# # -----------------------------
# def load_and_preprocess(img_path, target_size=(32, 32)):
#     img = cv2.imread(img_path)
#     if img is None:
#         raise FileNotFoundError(f"Image not found: {img_path}")
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, target_size)
#     img = img.astype("float32") / 255.0
#     return img

# images, y_true = [], []

# for idx, row in df_test.iterrows():
#     try:
#         img = load_and_preprocess(row['Path'])
#         images.append(img)
#         y_true.append(int(row['ClassId']))
#     except FileNotFoundError:
#         print(f"[WARNING] Missing image: {row['Path']}")

# images = np.array(images)
# y_true = np.array(y_true)
# print(f"[INFO] Loaded {len(images)} images for testing.")

# # -----------------------------
# # 6. Run batch prediction
# # -----------------------------
# preds = model.predict(images, batch_size=64, verbose=1)
# y_pred = np.argmax(preds, axis=1)

# # -----------------------------
# # 7. Calculate accuracy
# # -----------------------------
# accuracy = (y_pred == y_true).mean() * 100
# print(f"\n[RESULT] Model Accuracy on Test Set: {accuracy:.2f}%")

# # -----------------------------
# # 8. Confusion Matrix
# # -----------------------------
# plt.figure(figsize=(15, 12))
# ConfusionMatrixDisplay.from_predictions(
#     y_true, y_pred, display_labels=range(43),
#     cmap='coolwarm', normalize='true', xticks_rotation=90
# )
# plt.title("Normalized Confusion Matrix - Traffic Sign Recognition (GTSRB)")
# plt.show()

# # -----------------------------
# # 9. Classification Report
# # -----------------------------
# print("\nDetailed Classification Report:\n")
# print(classification_report(y_true, y_pred, target_names=class_names))



import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

# -----------------------------
# 0. Environment Setup
# -----------------------------
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Ensure GPU memory growth (optional but recommended)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("[INFO] GPU memory growth enabled.")
    except Exception as e:
        print("[WARNING] Could not set memory growth:", e)

# -----------------------------
# 1. Load trained model
# -----------------------------
MODEL_PATH = r"D:\CGC\PHD\conf\paper\traffic_sign_detection_gtsrb\models\resnet_finetuned.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = load_model(MODEL_PATH)
print("[INFO] Model loaded successfully!")

# -----------------------------
# 2. Load training CSV
# -----------------------------
TRAIN_CSV = r"C:\Users\GURPREET\Desktop\traffic_sign_detection_gtsrb\archive\Train.csv"
df_train = pd.read_csv(TRAIN_CSV)
num_train_images = len(df_train)
print(f"[INFO] Total training images: {num_train_images}")

# -----------------------------
# 3. Load or Generate Test CSV
# -----------------------------
TEST_CSV = r"C:\Users\GURPREET\Desktop\traffic_sign_detection_gtsrb\archive\Test.csv"
TEST_DIR = r"C:\Users\GURPREET\Desktop\traffic_sign_detection_gtsrb\archive\Test"

if not os.path.exists(TEST_CSV):
    print("[WARNING] Test.csv not found! Generating automatically...")
    data = []
    for class_id in os.listdir(TEST_DIR):
        class_folder = os.path.join(TEST_DIR, class_id)
        if os.path.isdir(class_folder):
            for img_name in os.listdir(class_folder):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join("Test", class_id, img_name)
                    data.append([img_path, int(class_id)])
    df_test = pd.DataFrame(data, columns=["Path", "ClassId"])
    df_test.to_csv(TEST_CSV, index=False)
    print(f"[INFO] Test.csv created successfully at: {TEST_CSV}")
else:
    df_test = pd.read_csv(TEST_CSV)
    print("[INFO] Loaded existing Test.csv")

# Fix relative paths if needed
df_test['Path'] = df_test['Path'].apply(lambda x: os.path.join(
    r"C:\Users\GURPREET\Desktop\traffic_sign_detection_gtsrb\archive", x))

num_test_images = len(df_test)
print(f"[INFO] Total test images: {num_test_images}")

# -----------------------------
# 4. Define 43 GTSRB class names
# -----------------------------
class_names = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)", "Speed limit (60km/h)",
    "Speed limit (70km/h)", "Speed limit (80km/h)", "End of speed limit (80km/h)", "Speed limit (100km/h)",
    "Speed limit (120km/h)", "No passing", "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection", "Priority road", "Yield", "Stop", "No vehicles",
    "Vehicles over 3.5 metric tons prohibited", "No entry", "General caution", "Dangerous curve to the left",
    "Dangerous curve to the right", "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing", "Bicycles crossing", "Beware of ice/snow",
    "Wild animals crossing", "End of all speed and passing limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left", "Keep right", "Keep left", "Roundabout mandatory",
    "End of no passing", "End of no passing by vehicles over 3.5 metric tons"
]

# -----------------------------
# 5. Load and preprocess test images
# -----------------------------
def load_and_preprocess(img_path, target_size=(96, 96)):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0
    return img

images, y_true = [], []

print("[INFO] Loading and preprocessing test images...")
for idx, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Processing"):
    try:
        img = load_and_preprocess(row['Path'])
        images.append(img)
        y_true.append(int(row['ClassId']))
    except FileNotFoundError:
        print(f"[WARNING] Missing image: {row['Path']}")

images = np.array(images)
y_true = np.array(y_true)
print(f"[INFO] Loaded {len(images)} valid test images.")

# -----------------------------
# 6. Run batch prediction
# -----------------------------
print("[INFO] Running model inference...")
preds = model.predict(images, batch_size=64, verbose=1)
y_pred = np.argmax(preds, axis=1)

# -----------------------------
# 7. Accuracy Calculation
# -----------------------------
accuracy = (y_pred == y_true).mean() * 100
print(f"\n✅ [RESULT] Model Accuracy on Test Set: {accuracy:.2f}%")

# -----------------------------
# 8. Confusion Matrix Visualization
# -----------------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(15, 12))
sns.heatmap(cm, annot=False, fmt='d', cmap='coolwarm')
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Confusion Matrix - Traffic Sign Recognition")
plt.tight_layout()
plt.show()

# -----------------------------
# 9. Classification Report
# -----------------------------
print("\n📊 Detailed Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# -----------------------------
# 10. Save Predictions to CSV
# -----------------------------
output_df = pd.DataFrame({
    "Path": df_test["Path"][:len(y_pred)],
    "True_Label": y_true,
    "Predicted_Label": y_pred,
    "Predicted_Class_Name": [class_names[i] for i in y_pred]
})
output_path = "test_predictions.csv"
output_df.to_csv(output_path, index=False)
print(f"[INFO] Predictions saved to: {os.path.abspath(output_path)}")






model = load_model(MODEL_PATH)
print("[INFO] Model loaded successfully!")










# import os
# import cv2
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import load_model
# from sklearn.metrics import classification_report, confusion_matrix
# import seaborn as sns

# # -----------------------------
# # 0. Suppress TensorFlow warnings
# # -----------------------------
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# # -----------------------------
# # 1. Define paths
# # -----------------------------
# BASE_DIR = r"d:\CGC\PHD\conf\paper\traffic_sign_detection_gtsrb"
# DATA_DIR = os.path.join(BASE_DIR, "data")
# MODEL_DIR = os.path.join(BASE_DIR, "models")
# OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# TEST_CSV = os.path.join(DATA_DIR, "Test", "Test.csv")
# MODEL_PATH = os.path.join(MODEL_DIR, "resnet_model.h5")

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # -----------------------------
# # 2. Check important files
# # -----------------------------
# if not os.path.exists(TEST_CSV):
#     raise FileNotFoundError(f"❌ Test.csv not found at {TEST_CSV}")

# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")

# print("✅ All paths verified.")

# # -----------------------------
# # 3. Constants
# # -----------------------------
# IMG_SIZE = (96, 96)     # MUST match training input
# BATCH_SIZE = 32

# # -----------------------------
# # 4. Load model
# # -----------------------------
# print("\n🧠 Loading trained model...")
# model = load_model(MODEL_PATH)
# print("✅ Model loaded successfully!")

# # -----------------------------
# # 5. Load test data paths
# # -----------------------------
# print("\n📂 Loading test image paths...")
# df_test = pd.read_csv(TEST_CSV)
# X_paths = df_test["Path"].values
# y_true = df_test["ClassId"].values

# print(f"✅ Found {len(X_paths)} test images.")

# # -----------------------------
# # 6. Memory-efficient batch prediction
# # -----------------------------
# def preprocess_image(img_path):
#     """Preprocess a single image for prediction."""
#     img = cv2.imread(img_path)
#     if img is None:
#         raise ValueError(f"Image not found: {img_path}")
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, IMG_SIZE)
#     img = img.astype(np.float32) / 255.0
#     return img

# def predict_in_batches(model, X_paths, batch_size):
#     """Predict in batches to avoid memory overflow."""
#     preds = []
#     print("\n🧠 Running memory-efficient prediction...")
#     for i in tqdm(range(0, len(X_paths), batch_size), desc="🚀 Predicting", ncols=100):
#         batch_paths = X_paths[i:i+batch_size]
#         batch_imgs = [preprocess_image(p) for p in batch_paths]
#         batch_imgs = np.array(batch_imgs, dtype=np.float32)
#         batch_pred = model.predict(batch_imgs, verbose=0)
#         preds.extend(np.argmax(batch_pred, axis=1))
#     return np.array(preds)

# # -----------------------------
# # 7. Run prediction
# # -----------------------------
# y_pred = predict_in_batches(model, X_paths, BATCH_SIZE)
# print("\n✅ Prediction complete!")

# # -----------------------------
# # 8. Evaluation
# # -----------------------------
# acc = np.mean(y_pred == y_true)
# print(f"\n🎯 Test Accuracy: {acc * 100:.2f}%")

# print("\n📊 Classification Report:")
# print(classification_report(y_true, y_pred, zero_division=0))

# # -----------------------------
# # 9. Confusion Matrix
# # -----------------------------
# print("\n📈 Generating confusion matrix...")
# cm = confusion_matrix(y_true, y_pred)
# plt.figure(figsize=(16, 12))
# sns.heatmap(cm, annot=False, cmap="viridis")
# plt.title("Traffic Sign Recognition - Confusion Matrix", fontsize=16)
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
# plt.close()
# print(f"✅ Confusion matrix saved at {OUTPUT_DIR}\\confusion_matrix.png")

# # -----------------------------
# # 10. Save Results
# # -----------------------------
# results_df = pd.DataFrame({
#     "ImagePath": X_paths,
#     "TrueLabel": y_true,
#     "PredictedLabel": y_pred
# })
# results_path = os.path.join(OUTPUT_DIR, "test_predictions.csv")
# results_df.to_csv(results_path, index=False)
# print(f"✅ Predictions saved at {results_path}")

# print("\n🎉 Evaluation Complete — Model Performance Summary Ready!")


















# # -----------------------------
# # 1B. Save Model Summary & Architecture
# # -----------------------------
# from tensorflow.keras.utils import plot_model
# from contextlib import redirect_stdout

# # 1️⃣ Print model summary in console
# print("\n📘 Model Architecture Summary:\n")
# model.summary()

# # 2️⃣ Save model summary to text file
# summary_path = "model_summary.txt"
# with open(summary_path, "w") as f:
#     with redirect_stdout(f):
#         model.summary()
# print(f"[INFO] Model summary saved to: {os.path.abspath(summary_path)}")

# # 3️⃣ Save model architecture diagram
# plot_model(model, to_file="model_structure.png", show_shapes=True, show_layer_names=True)
# print("[INFO] Model architecture diagram saved as: model_structure.png")
