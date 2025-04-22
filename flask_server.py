from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for
import os
from werkzeug.utils import secure_filename
import cv2
import pytesseract
import psycopg2
from datetime import datetime
import logging

#import WTForms for the web interface
from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SubmitField
from wtforms.validators import DataRequired, Optional, Length

#fuzzy wuzzy matching libraries
from fuzzywuzzy import process_image_ocr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='logs/server.log'
)
logger = logging.getLogger('flask_ocr_server')

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

app = Flask(__name__)


# Configure a secret key for Flask-WTF forms (REPLACE WITH A STRONG, UNIQUE KEY)
# Get from environment variable in production
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', '_,*H]wS;_ue|1SqXF~0B9~>HRYlDkr')

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('templates', exist_ok=True)  # Create templates directory if it doesn't exist
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Database configuration - MODIFY THESE WITH YOUR ACTUAL POSTGRESQL DETAILS
DB_CONFIG = {
    'dbname': 'ocr_data',
    'user': 'carlo',
    'password': 'pogi', # Replace with your actual password
    'host': 'localhost',
    'port': 5432
}
# --- NEW: Flask-WTF Form for Sorting Rules ---
class RuleForm(FlaskForm):
    # Add Length validator to match VARCHAR size in DB
    address_pattern = StringField('Address Pattern', validators=[DataRequired(), Length(max=255)])
    sorting_destination = StringField('Sorting Destination', validators=[DataRequired(), Length(max=255)])
    priority = IntegerField('Priority', validators=[Optional()], default=0)
    submit = SubmitField('Save Rule')
   
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def connect_to_db():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return None

def process_image_ocr(image_path):
    try:
        # Read image with OpenCV
        # Use IMREAD_COLOR to ensure 3 channels even if grayscale is needed later
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
             raise Exception(f"Could not open or find the image: {image_path}")

        # Basic preprocessing
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to enhance text
        # Consider adaptive thresholding for better results on varying lighting
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Added OTSU for automatic threshold

        # Save the preprocessed image (for debugging - optional)
        # preprocessed_path = image_path.replace('.', '_preprocessed.')
        # cv2.imwrite(preprocessed_path, thresh)
        # logger.info(f"Preprocessed image saved: {preprocessed_path}")

        # Perform OCR on the preprocessed image
        # Specify language if known, e.g., lang='eng'
        text = pytesseract.image_to_string(thresh)

        logger.info(f"OCR processing completed for {image_path}")
        return text.strip() # .strip() removes leading/trailing whitespace
    except Exception as e:
        logger.error(f"OCR processing error for {image_path}: {e}")
        raise

def save_to_database(text, image_path):
    conn = connect_to_db()
    if not conn:
        raise Exception("Failed to connect to database")

    try:
        cur = conn.cursor()

        # Insert OCR result into database
        # Ensure image_path stored is relative to UPLOAD_FOLDER or just the filename
        # Storing the full path might make serving them later harder if the server path changes
        # Let's store the filename relative to the UPLOAD_FOLDER
        relative_image_path = os.path.relpath(image_path, app.config['UPLOAD_FOLDER'])

        cur.execute(
            "INSERT INTO ocr_results (text_content, image_path, timestamp) VALUES (%s, %s, %s) RETURNING id",
            (text, relative_image_path, datetime.now())
        )

        # Get the ID of the inserted record
        result_id = cur.fetchone()[0]

        conn.commit()
        logger.info(f"OCR result saved to database with ID: {result_id} for image {relative_image_path}")

        return result_id
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error when saving {image_path}: {e}")
        raise
    finally:
        if conn:
            cur.close()
            conn.close()

# Create HTML template for the homepage
def create_index_template():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>OCR Results Viewer</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .upload-form {
                margin-bottom: 30px;
                padding: 20px;
                background-color: #f9f9f9;
                border-radius: 8px;
            }
            .results {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 20px;
            }
            .result-card {
                border: 1px solid #ddd;
                border-radius: 8px;
                overflow: hidden;
                background-color: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                display: flex;
                flex-direction: column; /* Stack content vertically */
            }
            .result-image {
                width: 100%;
                height: 200px; /* Fixed height for consistency */
                object-fit: contain; /* Keep aspect ratio, fit within bounds */
                background-color: #f0f0f0;
                border-bottom: 1px solid #eee; /* Separator */
            }
            .result-text {
                padding: 15px;
                flex-grow: 1; /* Allow text area to fill available space */
                overflow-y: auto;
                max-height: 150px; /* Limit text height but allow scrolling */
                font-size: 0.9em;
                line-height: 1.4;
            }
             .result-text pre {
                white-space: pre-wrap; /* Wrap long text */
                word-wrap: break-word;
            }
            .result-info {
                padding: 10px 15px;
                background-color: #f9f9f9;
                font-size: 0.8em; /* Slightly smaller font */
                color: #666;
                border-top: 1px solid #eee;
                text-align: right; /* Align info to the right */
            }
            input[type="file"] {
                margin: 10px 0;
                display: block; /* Make input a block element */
            }
            button {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 1em;
            }
            button:hover {
                background-color: #45a049;
            }
             .upload-form button {
                margin-top: 10px; /* Add space above button in form */
             }
            .empty-message {
                text-align: center;
                padding: 20px;
                color: #666;
                grid-column: 1 / -1; /* Span across all columns */
            }
            .refresh-btn {
                display: block;
                margin: 20px auto;
                padding: 10px 20px;
                background-color: #2196F3;
            }
            .refresh-btn:hover {
                background-color: #0b7dda;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>OCR Results Viewer</h1>

            <div class="upload-form">
                <h2>Upload New Image</h2>
                <form action="/upload_image" method="POST" enctype="multipart/form-data">
                    <input type="file" name="image" accept="image/png, image/jpeg">
                    <button type="submit">Upload and Process</button>
                </form>
            </div>

            <button class="refresh-btn" onclick="window.location.reload()">Refresh Results</button>

            <h2>Recent OCR Results</h2>
            <div class="results" id="results-container">
                {% if results %}
                    {% for result in results %}
                        <div class="result-card">
                            {# Construct URL for uploaded image #}
                            {% set image_url = '/uploads/' + result.image_path %}
                            <img class="result-image" src="{{ image_url }}" alt="OCR Image">
                            {# Use <pre> for preserving whitespace/newlines from OCR #}
                            <div class="result-text"><pre>{{ result.text }}</pre></div>
                            <div class="result-info">
                                ID: {{ result.id }} | Date: {{ result.timestamp.strftime('%Y-%m-%d %H:%M:%S') }}
                            </div>
                        </div>
                    {% endfor %}
                {% else %}
                    <div class="empty-message">No OCR results found.</div>
                {% endif %}
            </div>
        </div>
    </body>
    </html>
    """

    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)

    # Write the template to a file
    with open('templates/index.html', 'w') as f:
        f.write(html_content)

# Homepage route
@app.route('/')
def index():
    create_index_template()  # Ensure the template exists

    # Get results from database
    conn = connect_to_db()
    if not conn:
        # Log database connection error in the response context as well
        logger.error("Failed to connect to database for homepage view.")
        # Pass an error message to the template or handle in template
        return render_template('index.html', results=[], app_config=app.config, db_error=True)

    try:
        cur = conn.cursor()
        # Select relative path for serving
        cur.execute("SELECT id, text_content, image_path, timestamp FROM ocr_results ORDER BY timestamp DESC")
        results_data = cur.fetchall()

        results = []
        for row in results_data:
            results.append({
                'id': row[0],
                'text': row[1],
                'image_path': row[2], # This is now the relative path/filename
                'timestamp': row[3]
            })

        return render_template('index.html', results=results, app_config=app.config)
    except Exception as e:
        logger.error(f"Error fetching results for homepage: {e}")
        return render_template('index.html', results=[], app_config=app.config, fetch_error=True)
    finally:
        if conn:
            cur.close()
            conn.close()

# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # Ensure the filename is secure before joining paths
    safe_filename = secure_filename(filename)
    # Only serve files from the configured UPLOAD_FOLDER
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], safe_filename)
    except FileNotFoundError:
        logger.warning(f"Requested file not found in uploads: {safe_filename}")
        return "File not found", 404
    except Exception as e:
        logger.error(f"Error serving file {safe_filename}: {e}")
        return "Error serving file", 500


@app.route('/upload_image', methods=['POST'])
def upload_image():
    # Check if the post request has the file part
    if 'image' not in request.files:
        logger.warning("/upload_image: No 'image' part in the request")
        return jsonify({'error': 'No image part'}), 400

    file = request.files['image']

    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        logger.warning("/upload_image: No selected file")
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        try:
            # Save the file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Prevent overwriting if file exists (optional, depends on desired behavior)
            if os.path.exists(filepath):
                 base, ext = os.path.splitext(filename)
                 timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
                 filename = f"{base}_{timestamp_str}{ext}"
                 filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                 logger.info(f"File exists, saving as: {filename}")

            file.save(filepath)
            logger.info(f"Image saved: {filepath}")

            # Process the image with OCR
            text = process_image_ocr(filepath)

            # Save results to database
            result_id = save_to_database(text, filepath)

            # Always return JSON for the API endpoint
            return jsonify({
                'success': True,
                'message': f'Image {filename} uploaded and processed successfully',
                'result_id': result_id,
                'text': text
            })

        except Exception as e:
            logger.error(f"/upload_image: Error processing upload: {e}")
            # Make sure error responses are also JSON
            return jsonify({'error': f'Server error during processing: {str(e)}'}), 500
    else:
        logger.warning(f"/upload_image: Invalid file type received: {file.filename}")
        # Make sure invalid file type response is also JSON
        return jsonify({'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

@app.route('/health', methods=['GET'])
def health_check():
    # Simple health check endpoint
    # Could also check database connection here
    db_status = "ok"
    try:
        conn = connect_to_db()
        if conn:
            conn.close()
        else:
            db_status = "database connection failed"
    except Exception:
         db_status = "database connection failed"

    return jsonify({'status': 'ok', 'database': db_status})

@app.route('/results/<int:result_id>', methods=['GET'])
def get_result(result_id):
    # Retrieve a specific OCR result from the database
    conn = connect_to_db()
    if not conn:
        logger.error("/results/<id>: Database connection failed")
        return jsonify({'error': 'Database connection failed'}), 500

    try:
        cur = conn.cursor()
        # Select relative path for serving
        cur.execute("SELECT id, text_content, image_path, timestamp FROM ocr_results WHERE id = %s", (result_id,))
        result = cur.fetchone()

        if result:
            return jsonify({
                'id': result[0],
                'text': result[1],
                'image_path': result[2], # This is the relative path/filename
                'timestamp': result[3].isoformat() # Use ISO format for better compatibility
            })
        else:
            logger.warning(f"/results/<id>: Result ID {result_id} not found")
            return jsonify({'error': 'Result not found'}), 404
    except Exception as e:
        logger.error(f"/results/<id>: Database query error for ID {result_id}: {e}")
        return jsonify({'error': f'Server error retrieving result: {str(e)}'}), 500
    finally:
        if conn:
            cur.close()
            conn.close()

@app.route('/results', methods=['GET'])
def get_all_results():
    # Retrieve all OCR results from the database
    conn = connect_to_db()
    if not conn:
        logger.error("/results: Database connection failed")
        return jsonify({'error': 'Database connection failed'}), 500

    try:
        cur = conn.cursor()
        # Select relative path for serving
        cur.execute("SELECT id, text_content, image_path, timestamp FROM ocr_results ORDER BY timestamp DESC")
        results = cur.fetchall()

        result_list = []
        for row in results:
            result_list.append({
                'id': row[0],
                'text': row[1],
                'image_path': row[2], # This is the relative path/filename
                'timestamp': row[3].isoformat() # Use ISO format
            })

        return jsonify(result_list)
    except Exception as e:
        logger.error(f"/results: Database query error: {e}")
        return jsonify({'error': f'Server error retrieving results: {str(e)}'}), 500
    finally:
        if conn:
            cur.close()
            conn.close()

if __name__ == '__main__':
    # Create the database table if it doesn't exist
    try:
        conn = connect_to_db()
        if conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ocr_results (
                    id SERIAL PRIMARY KEY,
                    text_content TEXT,
                    image_path VARCHAR(255), -- Store relative path or filename
                    timestamp TIMESTAMP
                )
            """)
            conn.commit()
            cur.close()
            conn.close()
            logger.info("Database table checked/created successfully")
        else:
            logger.warning("Couldn't connect to database to create table")
    except Exception as e:
        logger.error(f"Error creating database table: {e}")

    # Create the template file if it doesn't exist
    create_index_template()

    # Start the Flask server
    logger.info("Starting Flask server")
    # In a production environment, use a WSGI server like Gunicorn or uWSGI
    # For development, debug=True is fine.
    app.run(debug=True, host='0.0.0.0', port=5000)