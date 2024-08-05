from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# ดิกชันนารีสำหรับระบุคลาสที่เป็นไปได้ทั้งหมด
dic = {0: 'angry', 1: 'happy', 2: 'relaxed', 3: 'sad', 4: 'unknown emotion'}

# โหลดโมเดลที่ฝึกมาแล้วสำหรับตรวจจับอารมณ์สุนัข
emotion_model = load_model('Model333.h5')
emotion_model.make_predict_function()  # จำเป็นถ้าคุณใช้ TensorFlow backend

# โหลดโมเดลที่ฝึกมาแล้วสำหรับตรวจสอบว่าสุนัขหรือไม่
dog_detector_model = load_model('keras_model.h5')  # โมเดลนี้คุณต้องฝึกมาเองหรือดาวน์โหลดมา
dog_detector_model.make_predict_function()

def is_dog(img_path):
    i = image.load_img(img_path, target_size=(224, 224))  # เปลี่ยนขนาดเป็น 224, 224
    i = image.img_to_array(i) / 255.0
    i = np.expand_dims(i, axis=0)
    p = np.argmax(dog_detector_model.predict(i), axis=-1)
    return p[0] == 0  # แก้ไขให้ 0 คือสุนัข และ 1 คือไม่ใช่สุนัข

def predict_emotion(img_path):
    i = image.load_img(img_path, target_size=(224, 224))  # เปลี่ยนขนาดเป็น 224, 224
    i = image.img_to_array(i) / 255.0
    i = np.expand_dims(i, axis=0)
    p = np.argmax(emotion_model.predict(i), axis=-1)
    return dic.get(p[0], 'unknown emotion')

# กำหนดเส้นทาง
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/about")
def about_page():
    return "ยินดีต้อนรับสู่ Artificial Intelligence Hub! โปรดติดตามเพื่ออัพเดทเพิ่มเติม"

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename
        img.save(img_path)
        if is_dog(img_path):
            prediction = predict_emotion(img_path)
        else:
            prediction = 'unknown'
        return render_template("index.html", prediction=prediction, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
