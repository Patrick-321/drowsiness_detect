import cv2
import time
import tkinter as tk
from tkinter import Button, Label
from PIL import Image, ImageTk
from datetime import datetime
import os

class CaptureCameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drowsiness Data Capture")
        self.root.configure(bg="#f0f0f0")
        self.root.resizable(False, False)

        self.cap = cv2.VideoCapture(0)
        time.sleep(1.0)

        # Video label
        self.video_label = Label(root, bg="black")
        self.video_label.grid(row=1, column=0, columnspan=3, padx=10, pady=10)

        # Buttons
        self.btn_yawn = Button(root, text="Capture Yawning", width=20, command=lambda: self.capture_image("yawn"))
        self.btn_yawn.grid(row=0, column=0, padx=5, pady=5)

        self.btn_drowsy = Button(root, text="Capture Closed Eyes", width=20, command=lambda: self.capture_image("drowsy"))
        self.btn_drowsy.grid(row=0, column=1, padx=5, pady=5)

        self.btn_awake = Button(root, text="Capture Open Eyes", width=20, command=lambda: self.capture_image("awake"))
        self.btn_awake.grid(row=0, column=2, padx=5, pady=5)

        # Info label
        self.status_label = Label(root, text="Live Feed Running", bg="#f0f0f0", fg="green", font=("Arial", 10, "italic"))
        self.status_label.grid(row=2, column=0, columnspan=3, pady=(0, 10))

        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))  # consistent size
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.video_label.after(15, self.update_frame)

    def capture_image(self, label):
        ret, frame = self.cap.read()
        if ret:
            os.makedirs("database/untagged_images", exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"database/untagged_images/{label}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            self.status_label.configure(text=f"Saved: {filename}", fg="blue")
            print(f"[INFO] Image saved to '{filename}'")

root = tk.Tk()
app = CaptureCameraApp(root)
root.mainloop()
