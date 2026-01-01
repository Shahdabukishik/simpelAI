print("Starting image classification...")   
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
import numpy as np

# مجلد الصور
IMAGE_DIR = os.path.join(os.path.dirname(__file__), "images")

X = []
y = []

for filename in os.listdir(IMAGE_DIR):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        label = filename.split("_")[0]  # cat_1.jpg -> label = 'cat'
        img = imread(os.path.join(IMAGE_DIR, filename))
        img = rgb2gray(img)  # أبيض وأسود
        img = resize(img, (28, 28))  # تصغير لتوحيد الأبعاد
        X.append(img.flatten())
        y.append(label)


print("Number of images loaded:", len(X))
X = np.array(X)
y = np.array(y)

print("Number of images loaded:", len(X))
print("pp:", y)
# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# مقياس البيانات
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# نموذج AI بسيط
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# التنبؤ والدقة
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy*100:.2f}%")
