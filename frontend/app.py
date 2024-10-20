from flask import *
import os

app = Flask(__name__)

manga_folder = '/Users/kavyaagar/tidal hack/TIDAL-Hackathon-2024/frontend/manga'
allowed_extension = {'pdf','png','jpg'}

@app.route('/welcome')
def welcome():
    return render_template("welcome.html")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extension

@app.route('/')
def index():
    manga = os.listdir(manga_folder)
    manga_list = {m: os.listdir(os.path.join(manga_folder,m)) for m in manga}
    print("Home route accessed")
    return render_template('index.html', manga_list = manga_list)

@app.route('/read/<manga_name>/<chapter>')
def reader(manga_name, chapter):
    filepath = os.path.join(manga_folder, manga_name, chapter)
    if allowed_file(chapter):
        return render_template('reader.html', manga_name=manga_name, chapter=chapter)
    return "Invalid chapter or file not found.", 404

@app.route('/read/<manga_name>/<chapter>')
def serve_manga(manga_name, chapter):
    return send_from_directory(os.path.join(manga_folder, manga_name), chapter)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)