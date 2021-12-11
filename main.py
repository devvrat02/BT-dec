from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import pickle
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'Brain_Tumor/Testing/user_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == 'POST':
        f = request.files['user_image']
        f.save(os.path.join(
            app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))

    with open("valid", "rb") as file:
        model_check = pickle.load(file)

    with open("model", "rb") as file:
        model = pickle.load(file)

    target_image = app.config['UPLOAD_FOLDER'] + "/" + f.filename    
    try:
        img_check = cv2.imread(target_image, 0)
        img1_check = cv2.resize(img_check, (300, 400))
        img1_check = img1_check.reshape(1, -1)/255
    

        validity_check = model_check.predict(img1_check)

        if validity_check[0] == 'yes':
            try:
                img = cv2.imread(target_image, 0)
                img1 = cv2.resize(img, (200, 200))
                img1 = img1.reshape(1, -1)/255
            except Exception as e:
                print(str(e))

            plt.imsave('static/img/uploaded_cmap1.jpg', img, cmap = 'gist_ncar')
            plt.imsave('static/img/uploaded_cmap2.jpg', img, cmap = 'tab20c')
            plt.imsave('static/img/uploaded_cmap3.jpg', img, cmap = 'nipy_spectral')
            plt.imsave('static/img/uploaded_cmap4.jpg', img, cmap = 'hot')
            output = model.predict(img1)
            if output[0] == "no_tumor":
                pred_tumor = False
                pred_color = "result__text_tagG"
                pred_title = "No Tumor detected"
                prediction = "Everything looks fine, stay safe :)"
            elif output[0] == "pituitary_tumor":
                pred_tumor = True
                pred_color = "result__text_tagR"
                pred_title = "Tumor detected"
                prediction = "'Pituitary Tumor' detected, we suggest you to keep calm and consult the doctor as soon as possible :)"
            elif output[0] == "glioma_tumor":
                pred_tumor = True
                pred_color = "result__text_tagR"
                pred_title = "Tumor detected"
                prediction = "'Glioma Tumor' detected, we suggest you to keep calm and consult the doctor as soon as possible :)"
            elif output[0] == "meningioma_tumor":
                pred_tumor = True
                pred_color = "result__text_tagR"
                pred_title = "Tumor detected"
                prediction = "'Meningioma Tumor' detected, we suggest you to keep calm and consult the doctor as soon as possible :)"

        elif validity_check[0] == 'no':
            pred_tumor = False
            pred_color = "result__text_tagR"
            pred_title = "Invalid Image"
            prediction = "Looks like the image does not fulfill the requirements, please upload a valid MRI image!"
    except Exception as e:
        pred_tumor = False
        pred_color = "result__text_tagR"
        pred_title = "Invalid Image"
        prediction = "Looks like the image does not fulfill the requirements, please upload a valid MRI image!"
    return render_template("index.html", pred_tumor = pred_tumor, pred_color=pred_color, prediction_tag=pred_title, prediction_result=prediction, scrollToAnchor="workplace_id")


if __name__ == "__main__":
    app.run(debug=True)
