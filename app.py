from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# ดิกชันนารีสำหรับระบุคลาสที่เป็นไปได้ทั้งหมด
dic = {0 : 'angry', 1 : 'Happy', 2 : 'relaxed', 3 : 'Sad'}

# โหลดโมเดลที่ฝึกมาแล้ว
model = load_model('Model333.h5')
model.make_predict_function()  # จำเป็นถ้าคุณใช้ TensorFlow backend

def predict_label(img_path):
    i = image.load_img(img_path, target_size=(224, 224))  # เปลี่ยนขนาดเป็น 224, 224
    i = image.img_to_array(i) / 255.0
    i = np.expand_dims(i, axis=0)
    p = np.argmax(model.predict(i), axis=-1)
    return dic[p[0]]

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
        p = predict_label(img_path)
        return render_template("index.html", prediction=p, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)


	