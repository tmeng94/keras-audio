from flask import Flask, request, jsonify
import subprocess
import os

from keras_audio.library.utility.gtzan_loader import gtzan_labels, gtzan_categories

app = Flask(__name__)

def predict(audio_path, model_dir_path):
    output = subprocess.check_output(['python3', 'predict_audio_category.py', audio_path, model_dir_path])
    print(output)
    return output.split(b'\n')[-2].decode("utf-8") 

@app.route('/hackaton/audio/categories', methods=['GET'])
def hello():
    assetPath = request.args.get('asset_path')

    path = os.path.join(os.path.expanduser('~'), 'src', assetPath.strip('/'))
    print('Path: ' + path)

    category_model_path = 'models_music_speech_soundeffect'
    genre_model_path = 'models_genres'

    category_predicted = int(predict(path, category_model_path))

    result = {
        'category': category_predicted
    }

    if result['category'] == 0:
        result['genre'] = int(predict(path, genre_model_path))
        
    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
