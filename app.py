from flask import Flask, request, render_template, jsonify
from deepface import DeepFace
import os
import uuid
from PIL import Image
import pillow_heif # Pastikan pillow_heif diimpor

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

def analyze_facial_attributes(img_path):
    """
    Analyze facial attributes including age, gender, race, and emotion.
    Returns formatted results or None if analysis fails.
    """
    try:
        print(f"Analyzing facial attributes for: {img_path}")
        result = DeepFace.analyze(
            img_path=img_path, 
            actions=['age', 'gender', 'race', 'emotion'],
            enforce_detection=False,
            detector_backend='retinaface'
        )
          # DeepFace.analyze returns a list of results (one per face detected)
        if isinstance(result, list) and len(result) > 0:
            analysis = result[0]  # Take the first face found
        else:
            analysis = result
        
        # Convert numpy values to regular Python types for JSON serialization
        def convert_scores(scores_dict):
            if not scores_dict:
                return {}
            return {k: float(v) for k, v in scores_dict.items()}
        
        # Format the results
        formatted_result = {
            'age': int(analysis.get('age', 0)),
            'gender': str(analysis.get('dominant_gender', 'Unknown')),
            'race': str(analysis.get('dominant_race', 'Unknown')),
            'emotion': str(analysis.get('dominant_emotion', 'Unknown')),
            'emotion_scores': convert_scores(analysis.get('emotion', {})),
            'race_scores': convert_scores(analysis.get('race', {})),
            'gender_scores': convert_scores(analysis.get('gender', {}))
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