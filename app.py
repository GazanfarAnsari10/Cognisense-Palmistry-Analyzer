from flask import Flask, render_template, request, jsonify, url_for
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import json
from datetime import datetime

from utils.hand_detection import HandDetector
from utils.line_extraction import LineExtractor
from utils.palmistry_rules import PalmistryAnalyzer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Generate unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{secure_filename(file.filename)}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the uploaded file
            file.save(filepath)
            
            # Process the image
            result = analyze_palm(filepath)
            
            if result['success']:
                result['original_image'] = url_for('static', filename=f'uploads/{filename}')
                return jsonify(result)
            else:
                # Clean up file if analysis failed
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify(result), 400
        
        return jsonify({'error': 'Invalid file type'}), 400
    
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

def analyze_palm(image_path):
    try:
        # Initialize components
        hand_detector = HandDetector()
        line_extractor = LineExtractor()
        palmistry_analyzer = PalmistryAnalyzer()
        
        # Load and process image
        image = cv2.imread(image_path)
        if image is None:
            return {'success': False, 'error': 'Could not load image'}
        
        # Detect hand landmarks
        landmarks, annotated_image = hand_detector.detect_hand(image)
        if landmarks is None:
            return {'success': False, 'error': 'No hand detected in image'}
        
        # Extract palm lines
        lines_info = line_extractor.extract_lines(image, landmarks)
        
        # Analyze palmistry
        analysis = palmistry_analyzer.analyze_palm(lines_info, landmarks)
        
        # Save annotated image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        annotated_filename = f"analyzed_{timestamp}.jpg"
        annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
        cv2.imwrite(annotated_path, annotated_image)
        
        return {
            'success': True,
            'analysis': analysis,
            'annotated_image': url_for('static', filename=f'uploads/{annotated_filename}'),
            'lines_detected': len(lines_info)
        }
    
    except Exception as e:
        return {'success': False, 'error': f'Analysis failed: {str(e)}'}

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)