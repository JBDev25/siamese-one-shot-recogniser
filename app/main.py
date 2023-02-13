# Kivy Dependencies
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock

from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Non Kivy Dependencies
import cv2
import os
import numpy as np
import tensorflow as tf
import time

from layers import L1Dist


# Build App and layout
class CamApp(App):

    def build(self):
        # Set logger

        # Establish Model and image dir
        self.model = tf.keras.models.load_model('./siam_model.h5', custom_objects={'L1Dist': L1Dist})
        self.input_image = os.path.join('application_data', 'input_image', 'input.jpg')
        self.verification_images = os.path.join('application_data', 'verification_images')

        # Main Layout
        self.webcam = Image(size_hint=(1, .8))
        self.verify_button = Button(text="Verify", size_hint=(1, .1), on_press=self.verify)
        self.capture_button = Button(text="Capture Anchor", size_hint=(1, .1), on_press=self.capture_anchor)
        self.verification_label = Label(size_hint=(1, .1), text=f"No Image Tests")

        # Items of layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.webcam)
        layout.add_widget(self.verify_button)
        layout.add_widget(self.verification_label)
        layout.add_widget(self.capture_button)

        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 33.0)
        return layout

    def update(self, *args):
        ret, frame = self.capture.read()
        frame = frame[100:100 + 250, 200:200 + 250, :]

        # Flip image and convert to texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.webcam.texture = img_texture

    def verify(self, *args):
        results = []
        # Set thresholds
        detection_threshold = 0.5
        verification_threshold = 0.5

        # Capture input image
        save_path = self.input_image
        ret, frame = self.capture.read()
        frame = frame[100:100 + 250, 200:200 + 250, :]
        cv2.imwrite(save_path, frame)

        # Get all verification images
        for image in os.listdir(self.verification_images):
            input_image = self.preprocess(self.input_image)
            ver_image = self.preprocess(os.path.join(self.verification_images, image))
            # Use model to classify
            res = self.model.predict(list(np.expand_dims([input_image, ver_image], axis=1)))
            results.append(res)
            print(res)
        print("="*10)
        # Detection theshold: Value over which is positive reuslt
        detection = np.sum(np.array(results) > detection_threshold)
        # Verification Threshold: Number of positived required to verify
        verifcation = detection / len(results)
        verified = verifcation > verification_threshold

        # Set Label to output
        self.verification_label.text = "Verified" if verified else "Not Verified"
        self.verification_label.color = 'green' if verified else 'red'

    def capture_anchor(self, *args):
        current_time = time.time()
        save_path = os.path.join(self.verification_images,f'{int(current_time)}.jpg')

        # Capture anchor image
        ret, frame = self.capture.read()
        frame = frame[100:100 + 250, 200:200 + 250, :]
        cv2.imwrite(save_path, frame)

        # Remove least oldest image if >10 images
        verification_images = os.listdir(self.verification_images)
        if len(verification_images) > 10:
            verification_images.sort()
            os.remove(os.path.join(self.verification_images,verification_images[0]))


    def preprocess(self, path):
        # Read Image from path
        byte_img = tf.io.read_file(path)
        # Load Image
        img = tf.io.decode_jpeg(byte_img)
        # Resize image to 100x100px
        img = tf.image.resize(img, (100, 100))
        # Scale px values to 0-1
        img = img / 255.0
        return img


if __name__ == "__main__":
    CamApp().run()
