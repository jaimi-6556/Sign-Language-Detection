import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import datetime
from keras.models import load_model
import numpy as np
import cv2

# Load model and class names
model = load_model("D:\sign language detection\sign_language_model.h5")
class_names = np.load("D:\sign language detection\class_names.npy")

def is_allowed_time():
    return 18 <= datetime.datetime.now().hour <= 22

def preprocess_image(img):
    img = cv2.resize(img, (64, 64))
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)




def predict_image(img):
    processed = preprocess_image(img)
    preds = model.predict(processed)
    return class_names[np.argmax(preds)]

class App:
    def __init__(self, root):
        root.title("Sign Language Detection")
        root.geometry("600x500")

        tk.Label(root, text="Sign Language Detection", font=("Arial", 20)).pack(pady=10)
        self.image_label = tk.Label(root)
        self.image_label.pack()
        self.result_label = tk.Label(root, text="", font=("Arial", 16))
        self.result_label.pack(pady=10)

        tk.Button(root, text="Upload Image", command=self.upload_image).pack(pady=5)
        tk.Button(root, text="Use Webcam", command=self.use_webcam).pack(pady=5)

    def upload_image(self):
        if not is_allowed_time():
            messagebox.showinfo("Time Restricted", "Only allowed from 6 PM to 10 PM.")
            return
        path = filedialog.askopenfilename()
        if path:
            img = cv2.imread(path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pred = predict_image(img_rgb)
            img_pil = Image.fromarray(img_rgb)
            self.image_label.configure(image=ImageTk.PhotoImage(img_pil.resize((300, 300))))
            self.image_label.image = ImageTk.PhotoImage(img_pil.resize((300, 300)))
            self.result_label.config(text=f"Prediction: {pred}")

    def use_webcam(self):
        if not is_allowed_time():
            messagebox.showinfo("Time Restricted", "Only allowed from 6 PM to 10 PM.")
            return
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Press Q to Capture", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        pred = predict_image(frame)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        self.image_label.configure(image=ImageTk.PhotoImage(img_pil.resize((300, 300))))
        self.image_label.image = ImageTk.PhotoImage(img_pil.resize((300, 300)))
        self.result_label.config(text=f"Prediction: {pred}")

root = tk.Tk()
app = App(root)
root.mainloop()
