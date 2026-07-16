import os
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# 🔹 Step 1: Paths
main_zip = r"C:\Users\JAWAD KHAN\Downloads\archive.zip"
extract_main = r"C:\Users\JAWAD KHAN\Downloads\nn\dataset"

# 🔹 Step 2: Extract main archive
if not os.path.exists(extract_main):
    print("📦 Extracting main archive...")
    with zipfile.ZipFile(main_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_main)
    print("✅ Main archive extracted.")
else:
    print("✅ Dataset folder already exists.")

# 🔹 Step 3: Locate image folder
data_folder = os.path.join(extract_main, "extracted_images")
if not os.path.exists(data_folder):
    raise FileNotFoundError(f"❌ Could not find extracted_images folder at {data_folder}")

print(f"✅ Found image folder: {data_folder}")

# 🔹 Step 4: Split automatically using ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2  # 80% train, 20% validation
)

train_gen = datagen.flow_from_directory(
    data_folder,
    target_size=(28, 28),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    data_folder,
    target_size=(28, 28),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# 🔹 Step 5: Build the CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(train_gen.num_classes, activation="softmax")
])

# 🔹 Step 6: Compile & Train
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print("\n🚀 Training started...\n")
history = model.fit(train_gen, validation_data=val_gen, epochs=10)

# 🔹 Step 7: Save model and labels
model.save("symbol_model.h5")
print("✅ Model saved as symbol_model.h5")

import json
with open("class_labels.json", "w") as f:
    json.dump(train_gen.class_indices, f)
print("✅ Class labels saved to class_labels.json")

# 🔹 Step 8: GPU info (optional)
print("\n💻 Available GPUs:", tf.config.list_physical_devices('GPU'))
import tkinter as tk
from PIL import Image, ImageOps, ImageGrab
import numpy as np
import tensorflow as tf
import json
import os

# ✅ Load trained model and label mappings
model = tf.keras.models.load_model("symbol_model.h5")

with open("class_labels.json", "r") as f:
    class_labels = json.load(f)
classes = {v: k for k, v in class_labels.items()}

# 🧠 Predict function
def predict_symbol():
    # Capture drawn area
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()
    img = ImageGrab.grab().crop((x, y, x1, y1))

    # Preprocess image
    img = img.convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Model prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    label_result.config(
        text=f"Prediction: {classes[predicted_class]} ({confidence*100:.2f}% confidence)"
    )

# ✏️ Drawing function
def draw(event):
    x, y = event.x, event.y
    r = 8  # Brush size
    canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")

# 🧹 Clear canvas
def clear_canvas():
    canvas.delete("all")
    label_result.config(text="Draw a symbol and click 'Predict'")

# 🪟 GUI Setup
root = tk.Tk()
root.title("🧠 Symbol Recognition")
root.geometry("400x480")
root.resizable(False, False)

canvas = tk.Canvas(root, width=280, height=280, bg="black", cursor="cross")
canvas.pack(pady=20)

label_result = tk.Label(root, text="Draw a symbol and click 'Predict'", font=("Arial", 14))
label_result.pack()

# 🔘 Buttons
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

tk.Button(
    btn_frame, text="Predict", command=predict_symbol, bg="#4CAF50", fg="white",
    font=("Arial", 12), width=10
).grid(row=0, column=0, padx=10)

tk.Button(
    btn_frame, text="Clear", command=clear_canvas, bg="#f44336", fg="white",
    font=("Arial", 12), width=10
).grid(row=0, column=1, padx=10)

canvas.bind("<B1-Motion>", draw)

root.mainloop()
