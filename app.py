from flask import Flask, request, render_template
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.models import Model
from keras.layers import Layer
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import UpSampling2D
from keras.layers import concatenate
from keras.layers import Add
from keras.layers import Multiply
from keras.layers import Input
from keras.layers import MaxPool2D
from keras.layers import BatchNormalization

app = Flask(__name__)
class EncoderBlock(Layer):

    def __init__(self, filters, rate, pooling=True, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)

        self.filters = filters
        self.rate = rate
        self.pooling = pooling

        self.c1 = Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')
        self.drop = Dropout(rate)
        self.c2 = Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')
        self.pool = MaxPool2D()

    def call(self, X):
        x = self.c1(X)
        x = self.drop(x)
        x = self.c2(x)
        if self.pooling:
            y = self.pool(x)
            return y, x
        else:
            return x

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters":self.filters,
            'rate':self.rate,
            'pooling':self.pooling
        }
class DecoderBlock(Layer):

    def __init__(self, filters, rate, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)

        self.filters = filters
        self.rate = rate

        self.up = UpSampling2D()
        self.net = EncoderBlock(filters, rate, pooling=False)

    def call(self, X):
        X, skip_X = X
        x = self.up(X)
        c_ = concatenate([x, skip_X])
        x = self.net(c_)
        return x

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters":self.filters,
            'rate':self.rate,
        }
class AttentionGate(Layer):

    def __init__(self, filters, bn, **kwargs):
        super(AttentionGate, self).__init__(**kwargs)

        self.filters = filters
        self.bn = bn

        self.normal = Conv2D(filters, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.down = Conv2D(filters, kernel_size=3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')
        self.learn = Conv2D(1, kernel_size=1, padding='same', activation='sigmoid')
        self.resample = UpSampling2D()
        self.BN = BatchNormalization()

    def call(self, X):
        X, skip_X = X

        x = self.normal(X)
        skip = self.down(skip_X)
        x = Add()([x, skip])
        x = self.learn(x)
        x = self.resample(x)
        f = Multiply()([x, skip_X])
        if self.bn:
            return self.BN(f)
        else:
            return f

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters":self.filters,
            "bn":self.bn
        }

classification_model = load_model("efficientnet_model.h5")  
segmentation_model = load_model("tumor_segmentation_model.h5", custom_objects={
    'EncoderBlock': EncoderBlock,
    'DecoderBlock': DecoderBlock,
    'AttentionGate': AttentionGate
})

def classify_image(image):
    image_resized = cv2.resize(image, (224, 224)) / 255.0  
    image_input = np.expand_dims(image_resized, axis=0)
    prediction = classification_model.predict(image_input)[0]
    classes = ["Benign", "Malignant", "Normal"]
    class_index = np.argmax(prediction)
    return classes[class_index], prediction[class_index] 


def segment_tumor(image):
    image_resized = cv2.resize(image, (256, 256)) / 255.0  
    image_input = np.expand_dims(image_resized, axis=0)
    prediction = segmentation_model.predict(image_input)[0]
    mask_resized = cv2.resize(prediction, (image.shape[1], image.shape[0]))
    mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
    overlay = image.copy()
    overlay[mask_binary == 255] = [255, 0, 0]  
    return overlay

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        filename = "static/input.png"
        file.save(filename)
        
        image = cv2.imread(filename)
        classification_result, confidence = classify_image(image)
        output_path = None

        if classification_result == "Normal":
            return render_template("index.html", input_image=filename, output_image=None, classification=classification_result, confidence=confidence)
        else:
            segmented_image = segment_tumor(image)
            output_path = "static/output.png"
            cv2.imwrite(output_path, segmented_image)
        
        return render_template("index.html", input_image=filename, output_image=output_path, classification=classification_result, confidence=confidence)
    
    return render_template("index.html", input_image=None, output_image=None, classification=None, confidence=None)

if __name__ == "__main__":
    app.run(debug=True)