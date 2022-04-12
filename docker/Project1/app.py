import os
from PIL import Image
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
from flask_uploads import UploadSet, configure_uploads, IMAGES
from predict import process, predict
from model import Origin


app = Flask(__name__)
app.config['SECRET_KEY']= os.urandom(24)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app.config['UPLOADED_PHOTO'] = './static'

@app.route('/')
def upload_form():
	return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def upload():
	if request.method == 'POST':
		if request.files['photo'].filename == '':
			flash('請上傳圖片')
			return render_template('index.html')
		file = request.files['photo']
		if file and allowed_file(file.filename):
		# return render_template("index.html", uploaded_image=file.filename )
			file.save(os.path.join(app.config['UPLOADED_PHOTO'], file.filename))
			filename = os.path.join(app.config['UPLOADED_PHOTO'], file.filename)
			image = Image.open(filename)
			image = process(image)
			label, prob = predict(image)

			flash("圖片: {}".format(file.filename))
			flash("判斷結果為 : {}".format(label))
			flash("判斷機率為 : {}%".format(prob))
			# return answer
			return render_template('index.html', filename=filename)
		else:
			flash('Allowed image types are -> png, jpg, jpeg, gif')
			return redirect(request.url)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8888)