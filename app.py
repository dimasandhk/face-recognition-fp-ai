from flask import Flask, request, render_template, jsonify
from deepface import DeepFace
import os
import uuid
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

app = Flask(__name__)
UPLOAD_FOLDER = 'temp_uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def convert_heic_to_jpg(heic_path):
    try:
        img = Image.open(heic_path)
        jpg_filename = os.path.splitext(os.path.basename(heic_path))[0] + "_" + str(uuid.uuid4())[:4] + ".jpg"
        jpg_path = os.path.join(os.path.dirname(heic_path), jpg_filename)
        img.convert('RGB').save(jpg_path, "JPEG")
        return jpg_path
    except Exception as e:
        print(f"Error converting HEIC {heic_path}: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Please upload both images'}), 400

    file1 = request.files['image1']
    file2 = request.files['image2']

    if file1.filename == '' or file2.filename == '':
        return jsonify({'error': 'File name cannot be empty'}), 400

    original_filename1 = str(uuid.uuid4()) + os.path.splitext(file1.filename)[1]
    original_filename2 = str(uuid.uuid4()) + os.path.splitext(file2.filename)[1]

    img1_path_original = os.path.join(app.config['UPLOAD_FOLDER'], original_filename1)
    img2_path_original = os.path.join(app.config['UPLOAD_FOLDER'], original_filename2)

    file1.save(img1_path_original)
    file2.save(img2_path_original)

    img1_path_for_deepface = img1_path_original
    img2_path_for_deepface = img2_path_original
    
    files_to_delete = [img1_path_original, img2_path_original]

    if original_filename1.lower().endswith(('.heic', '.heif')):
        converted_path1 = convert_heic_to_jpg(img1_path_original)
        if converted_path1:
            img1_path_for_deepface = converted_path1
            files_to_delete.append(converted_path1)
        else:
            for f_path in files_to_delete:
                if os.path.exists(f_path): os.remove(f_path)
            return jsonify({'error': 'Failed to convert image 1 (HEIC).'}), 500
    
    if original_filename2.lower().endswith(('.heic', '.heif')):
        converted_path2 = convert_heic_to_jpg(img2_path_original)
        if converted_path2:
            img2_path_for_deepface = converted_path2
            files_to_delete.append(converted_path2)
        else:
            for f_path in files_to_delete:
                if os.path.exists(f_path): os.remove(f_path)
            return jsonify({'error': 'Failed to convert image 2 (HEIC).'}), 500

    try:
        result = DeepFace.verify(img1_path=img1_path_for_deepface,
                                 img2_path=img2_path_for_deepface,
                                 model_name="VGG-Face",
                                 enforce_detection=False)
        
        return jsonify({
            'verified': result['verified'],
            'distance': result['distance'],
            'threshold': result['threshold'],
            'model': result['model'],
            'similarity_metric': result['similarity_metric']
        })

    except Exception as e:
        if "Face could not be detected" in str(e) or "cannot be aligned" in str(e):
             return jsonify({'error': 'Face could not be detected in one or both images.'}), 500
        return jsonify({'error': f'An error occurred during processing: {str(e)}'}), 500
    finally:
        for f_path in files_to_delete:
            if os.path.exists(f_path):
                try:
                    os.remove(f_path)
                except Exception as e_del:
                    print(f"Error deleting file {f_path}: {e_del}")

@app.route('/webcam', methods=['POST'])
def webcam_predict():
    if 'reference_image' not in request.files:
        return jsonify({'error': 'Please upload a reference image'}), 400

    reference_image = request.files['reference_image']
    reference_filename = str(uuid.uuid4()) + os.path.splitext(reference_image.filename)[1]
    reference_image_path = os.path.join(app.config['UPLOAD_FOLDER'], reference_filename)
    reference_image.save(reference_image_path)

    # Here you would implement the logic to capture the webcam image and compare it with the reference image
    # For now, we will just return a success message
    return jsonify({'message': 'Webcam image processed successfully.'})

@app.route('/predict_webcam', methods=['POST'])
def predict_webcam():
    if 'reference' not in request.files or 'webcam' not in request.files:
        return jsonify({'error': 'Please upload both reference and webcam images'}), 400

    ref_file = request.files['reference']
    webcam_file = request.files['webcam']

    if ref_file.filename == '' or webcam_file.filename == '':
        return jsonify({'error': 'File name cannot be empty'}), 400

    ref_filename = str(uuid.uuid4()) + os.path.splitext(ref_file.filename)[1]
    webcam_filename = str(uuid.uuid4()) + os.path.splitext(webcam_file.filename)[1]

    ref_path = os.path.join(app.config['UPLOAD_FOLDER'], ref_filename)
    webcam_path = os.path.join(app.config['UPLOAD_FOLDER'], webcam_filename)

    ref_file.save(ref_path)
    webcam_file.save(webcam_path)

    files_to_delete = [ref_path, webcam_path]

    try:
        result = DeepFace.verify(
            img1_path=ref_path,
            img2_path=webcam_path,
            model_name="VGG-Face",
            enforce_detection=False
        )
        return jsonify({
            'verified': result['verified'],
            'distance': result['distance'],
            'threshold': result['threshold'],
            'model': result['model'],
            'similarity_metric': result['similarity_metric']
        })
    except Exception as e:
        if "Face could not be detected" in str(e) or "cannot be aligned" in str(e):
            return jsonify({'error': 'Face could not be detected in one or both images.'}), 500
        return jsonify({'error': f'An error occurred during processing: {str(e)}'}), 500
    finally:
        for f_path in files_to_delete:
            if os.path.exists(f_path):
                try:
                    os.remove(f_path)
                except Exception as e_del:
                    print(f"Error deleting file {f_path}: {e_del}")

if __name__ == '__main__':
    app.run(debug=True)