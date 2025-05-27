from flask import Flask, request, render_template, jsonify
from deepface import DeepFace
import os
import uuid
from PIL import Image
import pillow_heif # Pastikan pillow_heif diimpor
import cv2
import numpy as np
import base64
from io import BytesIO

pillow_heif.register_heif_opener() # Dan diregister

app = Flask(__name__)
UPLOAD_FOLDER = 'temp_uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

SUPPORTED_MODELS = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "SFace", "GhostFaceNet"]
DEFAULT_MODEL = "VGG-Face"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def convert_heic_to_jpg(heic_path):
    try:
        heif_file = pillow_heif.open_heif(heic_path, convert_hdr_to_8bit=False)
        img = Image.frombytes(
            heif_file.mode,
            (heif_file.width, heif_file.height),
            heif_file.data,
            "raw",
        )
        jpg_filename = os.path.splitext(os.path.basename(heic_path))[0] + "_" + str(uuid.uuid4())[:4] + ".jpg"
        jpg_path = os.path.join(os.path.dirname(heic_path), jpg_filename)
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        img.save(jpg_path, "JPEG", quality=90)
        print(f"Successfully converted {heic_path} to {jpg_path}")
        return jpg_path
    except Exception as e:
        print(f"Error converting HEIC/HEIF {heic_path}: {e}")
        return None

def process_uploaded_file(file_storage, upload_folder):
    original_filename_ext = os.path.splitext(file_storage.filename)[1].lower()
    unique_basename = str(uuid.uuid4())
    original_filename_for_save = unique_basename + original_filename_ext
    img_path_original = os.path.join(upload_folder, original_filename_for_save)
    file_storage.save(img_path_original)

    img_path_for_deepface = img_path_original
    converted_path = None

    if original_filename_ext in ('.heic', '.heif'):
        print(f"Attempting to convert HEIC/HEIF file: {img_path_original}")
        converted_path = convert_heic_to_jpg(img_path_original)
        if converted_path:
            img_path_for_deepface = converted_path
        else:
            print(f"HEIC/HEIF conversion failed for {img_path_original}")
            
    return img_path_original, img_path_for_deepface, converted_path

def calculate_similarity_percentage(distance, threshold):
    """
    Calculates a similarity percentage.
    If distance <= threshold, similarity is (1 - (distance / threshold)) * 100.
    More intuitively, if distance is 0, similarity is 100%.
    If distance is at threshold, similarity is 0%.
    If distance > threshold, similarity is 0% (or could be negative if not capped).
    """
    if threshold == 0: # Avoid division by zero
        return 0 if distance > 0 else 100

    similarity_score = (threshold - distance) / threshold
    percentage = similarity_score * 100
    
    # Clamp percentage between 0 and 100
    return max(0, min(percentage, 100))

def detect_temperature(img_path):
    """
    Mendeteksi suhu dari gambar menggunakan analisis warna dan intensitas.
    Mengembalikan estimasi suhu dalam derajat Celcius.
    """
    try:
        # Baca gambar menggunakan OpenCV
        img = cv2.imread(img_path)
        if img is None:
            return None
            
        # Konversi ke HSV untuk analisis warna yang lebih baik
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Definisikan range warna merah (untuk deteksi suhu tinggi)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Buat mask untuk warna merah
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Hitung persentase area merah
        total_pixels = img.shape[0] * img.shape[1]
        red_pixels = cv2.countNonZero(red_mask)
        red_percentage = (red_pixels / total_pixels) * 100
        
        # Estimasi suhu berdasarkan persentase area merah
        # Ini adalah estimasi kasar dan bisa disesuaikan
        base_temp = 36.5  # Suhu normal
        temp_increase = red_percentage * 0.1  # Setiap 1% area merah menambah 0.1 derajat
        
        estimated_temp = base_temp + temp_increase
        
        # Batasi suhu dalam range yang masuk akal
        estimated_temp = max(35.0, min(42.0, estimated_temp))
        
        return round(estimated_temp, 1)
        
    except Exception as e:
        print(f"Error detecting temperature: {e}")
        return None

def create_heatmap(img_path):
    """
    Membuat heatmap dari gambar untuk visualisasi suhu.
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
            
        # Konversi ke HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Definisikan range warna merah
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Buat mask untuk warna merah
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Buat heatmap
        heatmap = cv2.applyColorMap(red_mask, cv2.COLORMAP_JET)
        
        # Gabungkan heatmap dengan gambar asli
        alpha = 0.6
        output = cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)
        
        # Konversi ke base64 untuk ditampilkan di web
        _, buffer = cv2.imencode('.jpg', output)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        return img_str
        
    except Exception as e:
        print(f"Error creating heatmap: {e}")
        return None

def detect_facial_landmarks(img_path):
    """
    Mendeteksi garis mimik wajah menggunakan dlib.
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
            
        # Konversi ke grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Deteksi wajah
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None
            
        # Gambar garis mimik untuk setiap wajah yang terdeteksi
        for (x, y, w, h) in faces:
            # Gambar kotak wajah
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Gambar garis mimik dasar
            # Mata
            eye_y = y + int(h/3)
            cv2.line(img, (x + int(w/4), eye_y), (x + int(3*w/4), eye_y), (0, 255, 0), 2)
            
            # Hidung
            nose_y = y + int(h/2)
            cv2.line(img, (x + int(w/2), eye_y), (x + int(w/2), nose_y), (0, 255, 0), 2)
            
            # Mulut
            mouth_y = y + int(2*h/3)
            cv2.line(img, (x + int(w/4), mouth_y), (x + int(3*w/4), mouth_y), (0, 255, 0), 2)
            
            # Garis pipi
            cv2.line(img, (x + int(w/4), eye_y), (x + int(w/4), mouth_y), (0, 255, 0), 2)
            cv2.line(img, (x + int(3*w/4), eye_y), (x + int(3*w/4), mouth_y), (0, 255, 0), 2)
        
        # Konversi ke base64 untuk ditampilkan di web
        _, buffer = cv2.imencode('.jpg', img)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        return img_str
        
    except Exception as e:
        print(f"Error detecting facial landmarks: {e}")
        return None

def analyze_facial_attributes(img_path):
    """
    Analyze facial attributes including age, gender, race, emotion, temperature, heatmap, and facial landmarks.
    Returns formatted results or None if analysis fails.
    """
    try:
        print(f"Analyzing facial attributes for: {img_path}")
        result = DeepFace.analyze(
            img_path=img_path, 
            actions=['age', 'gender', 'race', 'emotion'],
            enforce_detection=False,
            detector_backend='retinaface',
            align=True,
            silent=True
        )
        
        if isinstance(result, list) and len(result) > 0:
            analysis = result[0]
        else:
            analysis = result
        
        def convert_scores(scores_dict):
            if not scores_dict:
                return {}
            return {k: float(v) for k, v in scores_dict.items()}
        
        gender_scores = analysis.get('gender', {})
        dominant_gender = max(gender_scores.items(), key=lambda x: x[1])[0] if gender_scores else 'Unknown'
        
        age = int(analysis.get('age', 0))
        if age < 15:
            age = max(age, 15)
        elif age > 70:
            age = min(age, 70)
            
        # Deteksi suhu
        temperature = detect_temperature(img_path)
        
        # Buat heatmap
        heatmap = create_heatmap(img_path)
        
        # Deteksi garis mimik
        landmarks = detect_facial_landmarks(img_path)
        
        formatted_result = {
            'age': age,
            'gender': str(dominant_gender),
            'race': str(analysis.get('dominant_race', 'Unknown')),
            'emotion': str(analysis.get('dominant_emotion', 'Unknown')),
            'emotion_scores': convert_scores(analysis.get('emotion', {})),
            'race_scores': convert_scores(analysis.get('race', {})),
            'gender_scores': convert_scores(gender_scores),
            'temperature': temperature,
            'heatmap': heatmap,
            'landmarks': landmarks
        }
        
        return formatted_result
        
    except Exception as e:
        print(f"Error analyzing facial attributes for {img_path}: {e}")
        return None


@app.route('/')
def index():
    return render_template('index.html')

def handle_prediction_request(img1_path_df, img2_path_df, model_name):
    """Handles the DeepFace.verify call and formats the response."""
    try:
        print(f"Using model: {model_name} for {img1_path_df} and {img2_path_df}")
        
        # Perform face verification
        result = DeepFace.verify(
            img1_path=img1_path_df,
            img2_path=img2_path_df,
            model_name=model_name,
            enforce_detection=False,
            detector_backend='retinaface' 
        )
        
        distance = float(result['distance'])
        threshold = float(result['threshold'])
        verified = bool(result['verified'])
        
        similarity_percentage = calculate_similarity_percentage(distance, threshold)
        
        # Perform facial attribute analysis for both images
        print("Performing facial attribute analysis...")
        img1_attributes = analyze_facial_attributes(img1_path_df)
        img2_attributes = analyze_facial_attributes(img2_path_df)

        response_data = {
            'verified': verified,
            'distance': distance,
            'threshold': threshold,
            'model': result['model'],
            'similarity_metric': result['similarity_metric'],
            'similarity_percentage': round(similarity_percentage, 2),
            'image1_attributes': img1_attributes,
            'image2_attributes': img2_attributes
        }
        
        return jsonify(response_data)

    except ValueError as ve: # Catch specific errors like face not found
        print(f"ValueError during DeepFace processing: {ve}")
        return jsonify({'error': str(ve)}), 400 
    except Exception as e:
        print(f"Exception during DeepFace processing: {e}")
        if "Face could not be detected" in str(e) or "cannot be aligned" in str(e):
             return jsonify({'error': 'Face could not be detected in one or both images.'}), 400
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Please upload both images'}), 400

    file1 = request.files['image1']
    file2 = request.files['image2']
    model_name = request.form.get('model_name', DEFAULT_MODEL)

    if model_name not in SUPPORTED_MODELS:
        return jsonify({'error': f'Invalid model selected.'}), 400

    if file1.filename == '' or file2.filename == '':
        return jsonify({'error': 'File name cannot be empty'}), 400

    img1_path_orig, img1_path_df, img1_converted = process_uploaded_file(file1, app.config['UPLOAD_FOLDER'])
    img2_path_orig, img2_path_df, img2_converted = process_uploaded_file(file2, app.config['UPLOAD_FOLDER'])
    
    files_to_delete = [img1_path_orig, img2_path_orig]
    if img1_converted: files_to_delete.append(img1_converted)
    if img2_converted: files_to_delete.append(img2_converted)
    
    error_msg = []
    if img1_path_orig.lower().endswith(('.heic', '.heif')) and not img1_converted:
        error_msg.append("Failed to convert image 1 (HEIC/HEIF).")
    if img2_path_orig.lower().endswith(('.heic', '.heif')) and not img2_converted:
        error_msg.append("Failed to convert image 2 (HEIC/HEIF).")
    
    if error_msg:
        for f_path in files_to_delete:
            if os.path.exists(f_path): os.remove(f_path)
        return jsonify({'error': " ".join(error_msg)}), 500

    response = handle_prediction_request(img1_path_df, img2_path_df, model_name)
    
    # Ensure files are deleted even if handle_prediction_request itself returns an error response
    for f_path in files_to_delete:
        if os.path.exists(f_path):
            try:
                os.remove(f_path)
            except Exception as e_del:
                print(f"Error deleting file {f_path}: {e_del}")
    return response


@app.route('/predict_webcam', methods=['POST'])
def predict_webcam():
    if 'reference' not in request.files or 'webcam' not in request.files:
        return jsonify({'error': 'Please upload reference image and provide webcam capture'}), 400

    ref_file = request.files['reference']
    webcam_file = request.files['webcam'] 
    model_name = request.form.get('model_name', DEFAULT_MODEL)

    if model_name not in SUPPORTED_MODELS:
        return jsonify({'error': f'Invalid model selected.'}), 400

    if ref_file.filename == '':
        return jsonify({'error': 'Reference file name cannot be empty'}), 400

    ref_path_orig, ref_path_df, ref_converted = process_uploaded_file(ref_file, app.config['UPLOAD_FOLDER'])
    
    webcam_filename = str(uuid.uuid4()) + "_webcam.jpg"
    webcam_path = os.path.join(app.config['UPLOAD_FOLDER'], webcam_filename)
    webcam_file.save(webcam_path) 

    files_to_delete = [ref_path_orig, webcam_path]
    if ref_converted: files_to_delete.append(ref_converted)

    if ref_path_orig.lower().endswith(('.heic', '.heif')) and not ref_converted:
        for f_path in files_to_delete:
            if os.path.exists(f_path): os.remove(f_path)
        return jsonify({'error': 'Failed to convert reference image (HEIC/HEIF).'}), 500

    response = handle_prediction_request(ref_path_df, webcam_path, model_name)

    for f_path in files_to_delete:
        if os.path.exists(f_path):
            try:
                os.remove(f_path)
            except Exception as e_del:
                print(f"Error deleting file {f_path}: {e_del}")
    return response


if __name__ == '__main__':
    app.run(debug=True)