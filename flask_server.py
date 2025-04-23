from flask import Flask, request, redirect, url_for, render_template, session, flash, jsonify, abort
import os
import psycopg2
import logging
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from flask import send_from_directory
import subprocess
from pathlib import Path
from PIL import Image
import pytesseract

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('flask_ocr_server')

# Create necessary directories
os.makedirs('templates', exist_ok=True)
os.makedirs('logs', exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', '_,*H]wS;_ue|1SqXF~0B9~>HRYlDkr')

# Directory configuration - consolidate to avoid confusion
PROCESSED_IMAGE_FOLDER = 'simulation/processed_images'
UPLOAD_FOLDER = 'static/uploads'  # For serving uploaded images
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_IMAGE_FOLDER'] = PROCESSED_IMAGE_FOLDER

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_IMAGE_FOLDER'], exist_ok=True)

# Other configuration
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:pogi@localhost/ocr_data'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Database configuration
DB_CONFIG = {
    'dbname': 'ocr_data',
    'user': 'carlo',
    'password': 'pogi',
    'host': 'localhost',
    'port': 5432
}

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# User model
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

    def __init__(self, username, password):
        self.username = username
        self.password = generate_password_hash(password)

def connect_to_db():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return None

# Initialize database tables and admin user
@app.before_request
def initialize_db():
    with app.app_context():
        # Create all tables
        db.create_all()

        # Check if admin user exists
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            # Create admin user with default password
            admin = User(username='admin', password='password')
            db.session.add(admin)
            db.session.commit()
            logger.info("Admin user created")

# Route: Authentication
@app.route('/', methods=['GET', 'POST'])
def index():
    # If user is already logged in, redirect to OCR images page
    if 'user_id' in session:
        return redirect(url_for('ocr_images'))
    
    # For GET requests, just show the login page
    if request.method == 'GET':
        return render_template('login.html')
    
    # For POST requests, process the login form
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        logger.info(f"Login attempt for user: {username}")
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            # Redirect to OCR images page after successful login
            return redirect(url_for('ocr_images'))
        else:
            flash('Invalid username or password')
            return render_template('login.html')

# Upload OCR result
@app.route('/upload_ocr_result', methods=['POST'])
def upload_ocr_result():
    if 'image' not in request.files:
        return "No image part", 400

    image = request.files['image']
    image_name = request.form.get('image_name')
    text_content = request.form.get('text')

    # Save OCR result to database
    conn = connect_to_db()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO ocr_results (text_content, image_path, timestamp, predicted_destination, match_score)
                VALUES (%s, %s, NOW(), %s, %s)
            """, (text_content, image_name, 'Unknown', 0))
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            logger.error(f"Error saving OCR result: {e}")
            return "Database error", 500

    return "Success", 200

# OCR Images page
@app.route('/ocr-images', methods=['GET'])
def ocr_images():
    # Get OCR results
    results = get_ocr_results()
    
    # Return JSON if requested via AJAX
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify(results)
    
    # Otherwise render the template with results
    return render_template('ocr_images.html', results=results)

# Process new uploaded image
@app.route('/process_image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['PROCESSED_IMAGE_FOLDER'], filename)
        file.save(filepath)
        
        # Call your image_watcher.py or its function
        try:
            result = subprocess.run(['python', 'image_watcher.py', filepath], 
                                   capture_output=True, text=True)
            
            # For image_watcher.py, you'll need to make sure it accepts a file path parameter
            # and performs OCR on that specific file rather than watching a directory
            
            # You could also directly call OCR here if preferred:
            text = pytesseract.image_to_string(Image.open(filepath))
            
            # Save OCR result to database
            conn = connect_to_db()
            if conn:
                try:
                    cur = conn.cursor()
                    cur.execute("""
                        INSERT INTO ocr_results (text_content, image_path, timestamp, predicted_destination, match_score)
                        VALUES (%s, %s, NOW(), %s, %s)
                    """, (text, filename, 'Unknown', 0))
                    conn.commit()
                    cur.close()
                    conn.close()
                except Exception as e:
                    logger.error(f"Error saving OCR result: {e}")
            
            return jsonify({
                'success': True, 
                'filename': filename,
                'text': text
            }), 200
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return jsonify({'error': 'Processing failed'}), 500
    
    return jsonify({'error': 'Invalid request'}), 400

# Get processed images with metadata
def get_processed_images():
    images_dir = Path(app.config['PROCESSED_IMAGE_FOLDER'])
    images = []
    
    if images_dir.exists():
        for img_file in images_dir.glob('*.jpg'):
            # You could read associated metadata from a database or text file
            metadata = {
                'filename': img_file.name,
                'path': f"{app.config['PROCESSED_IMAGE_FOLDER']}/{img_file.name}",
                'tracking': 'TRACKING123456',
                'recipient': 'John Doe',
                'address': '123 Main St, Anytown, USA',
            }
            images.append(metadata)
    
    return images

# Get OCR results
def get_ocr_results():
    results = []
    folder = app.config['PROCESSED_IMAGE_FOLDER']
    
    # Get files from the processed images folder
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            full_path = os.path.join(folder, fname)
            
            # Try to get text from database first
            conn = connect_to_db()
            text = None
            
            if conn:
                try:
                    cur = conn.cursor()
                    cur.execute("SELECT text_content FROM ocr_results WHERE image_path = %s", (fname,))
                    result = cur.fetchone()
                    if result:
                        text = result[0]
                    cur.close()
                    conn.close()
                except Exception as e:
                    logger.error(f"Database error: {e}")
            
            # If not found in database, do OCR
            if not text:
                try:
                    text = pytesseract.image_to_string(Image.open(full_path))
                except Exception as e:
                    logger.error(f"OCR error for {fname}: {e}")
                    text = "Error processing image"
            
            results.append({'filename': fname, 'text': text})
    
    return results

# Dashboard page
@app.route('/dashboard', methods=['GET'])
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    
    username = session.get('username', 'User')
    return render_template('dashboard.html', username=username)

# Logout
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# Serve processed images
@app.route('/simulation/processed_images/<path:filename>')
def serve_simulation_images(filename):
    try:
        return send_from_directory(app.config['PROCESSED_IMAGE_FOLDER'], filename)
    except FileNotFoundError:
        abort(404)

# Serve uploaded images
@app.route('/static/uploads/<path:filename>')
def serve_uploaded_images(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except FileNotFoundError:
        abort(404)
#TRACKING PANEL SEARCH FUNCTION
@app.route('/track_package', methods=['POST'])
def track_package():
    tracking_number = request.form.get('tracking_number')
    recipient = request.form.get('recipient')
    date = request.form.get('date')
    address = request.form.get('address')
    
    conn = connect_to_db
    results = []
    
    if conn:
        try:
            cur = conn.cursor()
            #BUILD QUERY BASED ON PARAMETERS
            query = "SELECT * FROM ocr_results WHERE 1=1"
            params = []

            if tracking_number:
                query += " AND text_content ILIKE %s"
                params.append(f'%{tracking_number}%')

            if recipient:
                query += " AND text_content ILIKE %s"
                params.append(f'%{recipient}%')

            if date:
                query += " AND text_content ILIKE %s"
                params.append(f'%{date}%')
           

            if address:
                query += " AND text_content ILIKE %s"
                params.append(f'%{address}%')

            # Add order by timestamp
            query += " ORDER BY timestamp DESC LIMIT 10"
                        # Transform results for JSON response
            for row in db_results:
                # Adjust column indices based on your actual table structure
                results.append({
                    'id': row[0],
                    'text_content': row[1],
                    'image_path': row[2],
                    'timestamp': str(row[3]) if row[3] else None,
                    'predicted_destination': row[4] if len(row) > 4 else None,
                    'match_score': row[5] if len(row) > 5 else None
                })
                
            cur.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error tracking package: {e}")
            return jsonify({'error': 'Database error', 'details': str(e)}), 500
    
    return jsonify({'results': results})

if __name__ == '__main__':
    logger.info("Starting Flask server")
    app.run(debug=True, host='0.0.0.0', port=5000)