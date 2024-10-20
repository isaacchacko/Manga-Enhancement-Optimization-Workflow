from flask import *
import os

app = Flask(__name__)

manga_folder = '/Users/kavyaagar/tidal hack/TIDAL-Hackathon-2024/frontend/manga'
allowed_extension = {'pdf','png','jpg'}

manga_data = {
    "Naruto": {1: ["Chapter 1", "Chapter 2"], 2: ["Chapter 1"]},
    "One Piece": {1: ["Chapter 1"], 2: ["Chapter 1", "Chapter 2", "Chapter 3"]},
    "Vinland Saga": {1: ["Chapter 1"], 2: ["Chapter1", "Chapter 2"]}
}

@app.route('/')
def welcome():
    return render_template('welcome.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extension

@app.route('/')
def index():
    # List all available mangas (folders) in the manga directory
    try:
        manga = [m for m in os.listdir(manga_folder) if os.path.isdir(os.path.join(manga_folder, m))]
        # For each manga, list its volumes (also directories)
        # manga_list = {m: os.listdir(os.path.join(manga_folder, m)) for m in manga}
        manga_list = manga_data
    except FileNotFoundError:
        manga_list = {}  # Handle the case where the folder is missing

    # Pass the manga_list to the template
    return render_template('index.html', manga_list=manga_list)


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



@app.route('/select_manga', methods=['POST'])
def select_manga():
    selected_manga = request.form['manga']
    volumes = os.listdir(os.path.join(manga_folder, selected_manga))
    return render_template('volume_selection.html', manga=selected_manga, volumes=volumes)

def find_chapter(manga, volume, chapter):
    volume_dir = os.path.join(manga_folder, manga, f"Volume {volume}")
    chapter_file = os.path.join(volume_dir, f"{chapter}.txt")
    
    if os.path.exists(chapter_file):
        with open(chapter_file, 'r') as f:
            return f.read()
    else:
        return "Chapter not found."

@app.route('/view_chapter')
def view_chapter():
    manga = request.args.get('manga') 
    volume = request.args.get('volume')
    chapter = request.args.get('chapter')
    chapter_path = os.path.join(manga_folder, manga, volume, chapter, f"{chapter}.pdf")
                
    if not os.path.exists(chapter_path):
        return abort(404, description="Chapter not found")

    # Serve the PDF
    return send_file(chapter_path)

if __name__ == '__main__':
    app.run(debug=True)