# ... existing imports ...
from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for
import os
from werkzeug.utils import secure_filename
import cv2
import pytesseract
import psycopg2
from datetime import datetime
import logging

# --- NEW IMPORTS FOR WEB INTERFACE AND SORTING LOGIC ---
from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SubmitField
from wtforms.validators import DataRequired, Optional, Length
from fuzzywuzzy import process # For fuzzy matching address patterns
import os # Import os for SECRET_KEY

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

# --- NEW: Secret Key for Flask-WTF Forms ---
# IMPORTANT: REPLACE THIS WITH A STRONG, UNIQUE KEY IN PRODUCTION
# It's best practice to get this from environment variables
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a_very_secret_key_for_your_app_change_this')

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('templates', exist_ok=True)  # Create templates directory if it doesn't exist
os.makedirs('logs', exist_ok=True) # Ensure logs directory exists
# Ensure templates/rules directory exists for the new templates
os.makedirs('templates/rules', exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Database configuration - MODIFY THESE WITH YOUR ACTUAL POSTGRESQL DETAILS
# It's better to use environment variables for credentials in production
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
    """Establishes a connection to the PostgreSQL database."""
    try:
        # Use the DB_CONFIG dictionary to connect
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return None

def process_image_ocr(image_path):
    """Performs OCR on the given image file."""
    try:
        # Read image with OpenCV
        # Use IMREAD_COLOR to ensure 3 channels even if grayscale is needed later
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
             raise Exception(f"Could not open or find the image: {image_path}")

        # Basic preprocessing (can be expanded)
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
        # Use config for better results if needed, e.g., config='--psm 6'
        text = pytesseract.image_to_string(thresh)

        logger.info(f"OCR processing completed for {image_path}")
        return text.strip() # .strip() removes leading/trailing whitespace
    except Exception as e:
        logger.error(f"OCR processing error for {image_path}: {e}")
        raise

# --- MODIFIED: save_to_database function to accept predicted_destination ---
def save_to_database(text, image_path, predicted_destination):
    """Saves the OCR text, image path, and predicted destination to the ocr_results table."""
    conn = connect_to_db()
    if not conn:
        raise Exception("Failed to connect to database")

    try:
        cur = conn.cursor()

        # Insert OCR result into database
        # Store the filename relative to the UPLOAD_FOLDER
        relative_image_path = os.path.relpath(image_path, app.config['UPLOAD_FOLDER'])

        # Include predicted_destination in the INSERT statement
        cur.execute(
            "INSERT INTO ocr_results (text_content, image_path, timestamp, predicted_destination) VALUES (%s, %s, %s, %s) RETURNING id",
            (text, relative_image_path, datetime.now(), predicted_destination)
        )

        # Get the ID of the inserted record
        result_id = cur.fetchone()[0]

        conn.commit()
        logger.info(f"OCR result saved to database with ID: {result_id} for image {relative_image_path}, Destination: {predicted_destination}")

        return result_id
    except Exception as e:
        conn.rollback() # Rollback changes if insertion fails
        logger.error(f"Database error when saving {image_path}: {e}")
        raise
    finally:
        if conn:
            cur.close()
            conn.close()

# Create HTML template for the homepage (existing function)
# This function generates the index.html file if it doesn't exist.
# You might want to create this file manually and put your desired HTML there.
# For this example, we keep the generation logic.
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
            /* --- MODIFIED: Styling for grouped results --- */
            .destination-group {
                margin-bottom: 30px;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 15px;
                background-color: #f9f9f9;
            }
            .destination-group h3 {
                margin-top: 0;
                color: #555;
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
                margin-bottom: 15px;
            }
            .results-grid {
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
                /* grid-column: 1 / -1; /* Span across all columns - might need adjustment with grouping */
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
            /* --- Added link to Rules Management --- */
            .rules-link {
                display: block;
                text-align: center;
                margin-top: 20px;
                font-size: 1.1em;
            }
             .rules-link a {
                color: #007bff;
                text-decoration: none;
             }
             .rules-link a:hover {
                text-decoration: underline;
             }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>OCR Results Viewer</h1>

            <div class="upload-form">
                <h2>Upload New Image</h2>
                <form action="{{ url_for('upload_image') }}" method="POST" enctype="multipart/form-data">
                    <input type="file" name="image" accept="image/png, image/jpeg">
                    <button type="submit">Upload and Process</button>
                </form>
            </div>

            <button class="refresh-btn" onclick="window.location.reload()">Refresh Results</button>

            <h2>Recent OCR Results by Destination</h2>

            {# --- MODIFIED: Loop through grouped results --- #}
            {% if grouped_results %}
                {% for destination, results_list in grouped_results.items() %}
                    <div class="destination-group">
                        <h3>Destination: {{ destination }}</h3>
                        <div class="results-grid">
                            {% for result in results_list %}
                                <div class="result-card">
                                    {# Construct URL for uploaded image #}
                                    {% set image_url = url_for('uploaded_file', filename=result.image_path) %}
                                    <img class="result-image" src="{{ image_url }}" alt="OCR Image">
                                    {# Use <pre> for preserving whitespace/newlines from OCR #}
                                    <div class="result-text"><pre>{{ result.text }}</pre></div>
                                    <div class="result-info">
                                        ID: {{ result.id }} | Date: {{ result.timestamp.strftime('%Y-%m-%d %H:%M:%S') }}
                                        {# Predicted: {{ result.predicted_destination }} #}
                                    </div>
                                </div>
                            {% endfor %}
                        </div>
                    </div>
                {% endfor %}
            {% else %}
                 <div class="empty-message">No OCR results found.</div>
            {% endif %}

            {# --- Added link to Rules Management --- #}
            <div class="rules-link">
                <a href="{{ url_for('list_rules') }}">Manage Sorting Rules</a>
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

# Homepage route (existing route)
@app.route('/')
def index():
    create_index_template()  # Ensure the template exists

    # Get results from database
    conn = connect_to_db()
    if not conn:
        # Log database connection error in the response context as well
        logger.error("Failed to connect to database for homepage view.")
        # Pass an error message to the template or handle in template
        return render_template('index.html', grouped_results={}, app_config=app.config, db_error=True) # Pass empty dict

    try:
        cur = conn.cursor()
        # --- MODIFIED: Select predicted_destination as well ---
        cur.execute("SELECT id, text_content, image_path, timestamp, predicted_destination FROM ocr_results ORDER BY predicted_destination, timestamp DESC")
        results_data = cur.fetchall()

        # --- NEW: Group results by predicted_destination ---
        grouped_results = {}
        for row in results_data:
            result = {
                'id': row[0],
                'text': row[1],
                'image_path': row[2], # This is now the relative path/filename
                'timestamp': row[3],
                'predicted_destination': row[4] # Get the predicted destination
            }
            destination = result['predicted_destination'] if result['predicted_destination'] else "Unknown/No Match" # Handle None destination
            if destination not in grouped_results:
                grouped_results[destination] = []
            grouped_results[destination].append(result)

        # --- MODIFIED: Pass grouped_results to the template ---
        return render_template('index.html', grouped_results=grouped_results, app_config=app.config)
    except Exception as e:
        logger.error(f"Error fetching results for homepage: {e}")
        return render_template('index.html', grouped_results={}, app_config=app.config, fetch_error=True) # Pass empty dict on error
    finally:
        if conn:
            cur.close()
            conn.close()

# Serve uploaded files (existing route)
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serves uploaded files from the UPLOAD_FOLDER."""
    # Ensure the filename is secure before joining paths
    safe_filename = secure_filename(filename)
    # Only serve files from the configured UPLOAD_FOLDER
    try:
        # Use send_from_directory to safely serve files
        return send_from_directory(app.config['UPLOAD_FOLDER'], safe_filename)
    except FileNotFoundError:
        logger.warning(f"Requested file not found in uploads: {safe_filename}")
        return "File not found", 404
    except Exception as e:
        logger.error(f"Error serving file {safe_filename}: {e}")
        return "Error serving file", 500


# --- MODIFIED: Image Upload and Processing Route ---
@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Handles image upload, performs OCR, looks up sorting rule, and saves results."""
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
            # Save the file securely
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Prevent overwriting if file exists (optional, depends on desired behavior)
            # Your code already handles this by adding a timestamp, which is good.
            if os.path.exists(filepath):
                 base, ext = os.path.splitext(filename)
                 timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S%f") # Added microseconds for uniqueness
                 filename = f"{base}_{timestamp_str}{ext}"
                 filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                 logger.info(f"File exists, saving as: {filename}")


            file.save(filepath)
            logger.info(f"Image saved: {filepath}")

            # Process the image with OCR
            text = process_image_ocr(filepath) # <-- OCR text is extracted here
            logger.info(f"OCR Extracted Text: {text}") # Log the extracted text

            # --- NEW: Lookup Sorting Rule from Database ---
            conn = None
            cursor = None
            predicted_destination = "NO MATCH" # Default if no rule matches
            match_score = 0
            matched_pattern = None
            # Define the fuzzy match threshold (adjust as needed)
            FUZZY_MATCH_THRESHOLD = 75 # Score out of 100

            try:
                conn = connect_to_db() # Use your existing DB connection function
                if conn:
                    cursor = conn.cursor()
                    # Fetch rules ordered by priority (highest first)
                    cursor.execute("SELECT address_pattern, sorting_destination, priority FROM sorting_rules ORDER BY priority DESC;")
                    rules = cursor.fetchall()

                    # Only attempt matching if rules exist and text is not empty/whitespace
                    if rules and text and text.strip():
                        logger.info(f"Found {len(rules)} sorting rules for lookup.")
                        # Normalize the extracted text for matching (basic example)
                        normalized_text = text.lower().strip()

                        # Create a list of just the patterns from the rules
                        patterns = [rule[0] for rule in rules]

                        # Use fuzzy matching to find the best rule match
                        # process.extractOne returns (best_match_string, score) or (best_match_string, score, index)
                        # depending on fuzzywuzzy version. (match, score) is consistent.
                        best_match = process.extractOne(normalized_text, patterns)

                        if best_match:
                            matched_pattern, match_score = best_match # Extract match and score

                            logger.info(f"Fuzzy match found: '{matched_pattern}' with score {match_score} for text fragment '{normalized_text[:50]}...'")

                            if match_score >= FUZZY_MATCH_THRESHOLD:
                                # Find the rule details in the original list based on the matched pattern string
                                # This is safer than relying on index if rule order from DB query changes
                                for rule in rules:
                                    if rule[0] == matched_pattern:
                                        predicted_destination = rule[1] # Get sorting_destination from the rule
                                        logger.info(f"Match above threshold ({FUZZY_MATCH_THRESHOLD}), predicted destination: {predicted_destination}")
                                        break # Stop once the rule is found
                                # If loop finishes without finding pattern (unlikely with extractOne match but safety check)
                                if predicted_destination == "NO MATCH" and match_score >= FUZZY_MATCH_THRESHOLD:
                                     logger.warning(f"Fuzzy match found pattern '{matched_pattern}' but couldn't find it in original rules list.")


                            else:
                                logger.info(f"Best match score ({match_score}) below threshold ({FUZZY_MATCH_THRESHOLD}). No sorting rule applied.")

                    elif not rules:
                         logger.warning("No sorting rules found in the database to perform lookup.")
                    # If text is empty, predicted_destination remains "NO MATCH" as set initially


                else:
                    logger.error("Failed to connect to database for sorting rule lookup.")
                    predicted_destination = "DB CONNECTION ERROR"
                    match_score = -1


            except Exception as db_lookup_err:
                logger.error(f"Error during sorting rule lookup from DB: {db_lookup_err}", exc_info=True)
                predicted_destination = "LOOKUP ERROR" # Indicate a lookup error occurred
                match_score = -1


            finally:
                if cursor:
                    cursor.close()
                if conn:
                    conn.close()
            # --- END NEW Lookup ---


            # Save original OCR results (text and image path) to the 'ocr_results' table
            # --- MODIFIED: Pass predicted_destination to save_to_database ---
            result_id = save_to_database(text, filepath, predicted_destination)
            logger.info(f"OCR result saved to database with ID: {result_id}")


            # --- MODIFIED: Return JSON response with sorting result ---
            # Return the determined destination and other info back to image_watcher.py
            return jsonify({
                'success': True,
                'message': f'Image {filename} uploaded and processed successfully',
                'result_id': result_id,
                'extracted_text': text, # Include the extracted text
                'predicted_destination': predicted_destination, # Include the determined destination
                'match_score': match_score # Include the match score from fuzzy matching
            })

        except Exception as e:
            logger.error(f"/upload_image: Error processing upload: {e}", exc_info=True)
            # Make sure error responses are also JSON
            return jsonify({'error': f'Server error during processing: {str(e)}'}), 500
    else:
        logger.warning(f"/upload_image: Invalid file type received: {file.filename}")
        # Make sure invalid file type response is also JSON
        return jsonify({'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

# --- NEW: Routes for Sorting Rules Web Interface ---

@app.route('/rules')
def list_rules():
    """Displays a list of all sorting rules."""
    conn = None
    cursor = None
    rules = []
    db_error = None
    try:
        conn = connect_to_db() # Use your existing DB connection function
        if conn:
            cursor = conn.cursor()
            # Select data from the sorting_rules table
            cursor.execute("SELECT id, address_pattern, sorting_destination, priority FROM sorting_rules ORDER BY priority DESC, address_pattern;")
            rules = cursor.fetchall()
        else:
            db_error = "Could not connect to database."

    except Exception as e:
        logger.error(f"Error fetching rules: {e}", exc_info=True)
        db_error = f"Error fetching rules: {e}"
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    # Pass rules and any potential error message to the template
    return render_template('rules/list.html', rules=rules, db_error=db_error)

@app.route('/rules/add', methods=['GET', 'POST'])
def add_rule():
    """Handles adding a new sorting rule."""
    form = RuleForm() # Use the form defined in Step 3
    if form.validate_on_submit():
        address_pattern = form.address_pattern.data
        sorting_destination = form.sorting_destination.data
        priority = form.priority.data if form.priority.data is not None else 0 # Handle Optional field

        conn = None
        cursor = None
        try:
            conn = connect_to_db() # Use your existing DB connection function
            if conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO sorting_rules (address_pattern, sorting_destination, priority) VALUES (%s, %s, %s);",
                    (address_pattern, sorting_destination, priority)
                )
                conn.commit()
                logger.info(f"Rule added: {address_pattern} -> {sorting_destination}")
                # Redirect to the list page after adding
                # flash("Rule added successfully!", "success") # Optional: use flash messages
                return redirect(url_for('list_rules'))
            else:
                 # flash("Database connection failed while adding rule.", "danger") # Optional
                 logger.error("Database connection failed for add_rule.")

        except Exception as e:
            logger.error(f"Error adding rule: {e}", exc_info=True)
            if conn: # Check if conn exists before rollback
                 conn.rollback() # Rollback changes on error
            # flash(f"Error adding rule: {e}", "danger") # Optional
            # Re-render the form with errors if database operation fails
            return render_template('rules/add_edit.html', form=form, title='Add Sorting Rule', db_error=f"Database error: {e}")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    # Render the add form template for GET requests or validation errors
    return render_template('rules/add_edit.html', form=form, title='Add Sorting Rule')

@app.route('/rules/edit/<int:rule_id>', methods=['GET', 'POST'])
def edit_rule(rule_id):
    """Handles editing an existing sorting rule."""
    conn = None
    cursor = None
    rule = None
    db_error = None
    try:
        conn = connect_to_db() # Use your existing DB connection function
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, address_pattern, sorting_destination, priority FROM sorting_rules WHERE id = %s;", (rule_id,))
            rule = cursor.fetchone()
        else:
            db_error = "Could not connect to database."

    except Exception as e:
        logger.error(f"Error fetching rule for edit (ID {rule_id}): {e}", exc_info=True)
        db_error = f"Error fetching rule: {e}"
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

    if not rule:
        # Rule not found, redirect to list or show 404
        # flash(f"Rule with ID {rule_id} not found.", "warning") # Optional
        return redirect(url_for('list_rules')) # Redirect to list if rule doesn't exist

    # Pass the rule data to the form for editing (only for GET request)
    form = RuleForm() # Create form instance

    if request.method == 'GET':
         # Populate form fields with data from the fetched rule
         form.address_pattern.data = rule[1]
         form.sorting_destination.data = rule[2]
         form.priority.data = rule[3] if rule[3] is not None else 0 # Handle potential None

    if form.validate_on_submit():
        # Process POST request with validated data
        address_pattern = form.address_pattern.data
        sorting_destination = form.sorting_destination.data
        priority = form.priority.data if form.priority.data is not None else 0

        conn = None
        cursor = None
        try:
            conn = connect_to_db() # Use your existing DB connection function
            if conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE sorting_rules SET address_pattern = %s, sorting_destination = %s, priority = %s WHERE id = %s;",
                    (address_pattern, sorting_destination, priority, rule_id)
                )
                conn.commit()
                logger.info(f"Rule updated: ID {rule_id}")
                # flash("Rule updated successfully!", "success") # Optional
                return redirect(url_for('list_rules'))
            else:
                 # flash("Database connection failed while updating rule.", "danger") # Optional
                 logger.error(f"Database connection failed for edit_rule ID {rule_id}.")
                 # Decide if you still want to redirect or return an error

        except Exception as e:
            logger.error(f"Error updating rule ID {rule_id}: {e}", exc_info=True)
            if conn:
                 conn.rollback() # Rollback on error
            # flash(f"Error updating rule: {e}", "danger") # Optional
            # Re-render the form with errors if database operation fails
            return render_template('rules/add_edit.html', form=form, title='Edit Sorting Rule', rule_id=rule_id, db_error=f"Database error: {e}")

        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    # Render the edit form template for GET requests or validation errors
    return render_template('rules/add_edit.html', form=form, title='Edit Sorting Rule', rule_id=rule_id, db_error=db_error)


@app.route('/rules/delete/<int:rule_id>', methods=['POST'])
def delete_rule(rule_id):
    """Handles deleting a sorting rule."""
    conn = None
    cursor = None
    try:
        conn = connect_to_db() # Use your existing DB connection function
        if conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sorting_rules WHERE id = %s;", (rule_id,))
            conn.commit()
            logger.info(f"Rule deleted: ID {rule_id}")
            # flash("Rule deleted successfully!", "success") # Optional
        else:
             # flash("Database connection failed while deleting rule.", "danger") # Optional
             logger.error(f"Database connection failed for delete_rule ID {rule_id}.")
             # Decide if you still want to redirect or return an error
             # For a POST delete, redirecting is common practice

    except Exception as e:
        logger.error(f"Error deleting rule ID {rule_id}: {e}", exc_info=True)
        if conn:
             conn.rollback() # Rollback on error
        # flash(f"Error deleting rule: {e}", "danger") # Optional
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    # Redirect back to the list page after deletion
    return redirect(url_for('list_rules'))

# --- END NEW Routes ---


# Existing routes for /health, /results/<id>, /results

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
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
    """Retrieves a specific OCR result from the database."""
    conn = connect_to_db()
    if not conn:
        logger.error("/results/<id>: Database connection failed")
        return jsonify({'error': 'Database connection failed'}), 500

    try:
        cur = conn.cursor()
        # --- MODIFIED: Select predicted_destination here too if needed for this endpoint ---
        cur.execute("SELECT id, text_content, image_path, timestamp, predicted_destination FROM ocr_results WHERE id = %s", (result_id,))
        result = cur.fetchone()

        if result:
            return jsonify({
                'id': result[0],
                'text': result[1],
                'image_path': result[2], # This is the relative path/filename
                'timestamp': result[3].isoformat(), # Use ISO format for better compatibility
                'predicted_destination': result[4] # Include predicted_destination
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
    """Retrieves all OCR results from the database."""
    conn = connect_to_db()
    if not conn:
        logger.error("/results: Database connection failed")
        return jsonify({'error': 'Database connection failed'}), 500

    try:
        cur = conn.cursor()
        # --- MODIFIED: Select predicted_destination here too ---
        cur.execute("SELECT id, text_content, image_path, timestamp, predicted_destination FROM ocr_results ORDER BY timestamp DESC")
        results = cur.fetchall()

        result_list = []
        for row in results:
            result_list.append({
                'id': row[0],
                'text': row[1],
                'image_path': row[2], # This is the relative path/filename
                'timestamp': row[3].isoformat(), # Use ISO format
                'predicted_destination': row[4] # Include predicted_destination
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
    # --- NEW: Create the sorting_rules database table if it doesn't exist ---
    try:
        conn = connect_to_db()
        if conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sorting_rules (
                    id SERIAL PRIMARY KEY,
                    address_pattern VARCHAR(255) UNIQUE NOT NULL,
                    sorting_destination VARCHAR(255) NOT NULL,
                    priority INTEGER DEFAULT 0,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            cur.close()
            conn.close()
            logger.info("Database table 'sorting_rules' checked/created successfully")
        else:
            logger.warning("Couldn't connect to database to create 'sorting_rules' table")
    except Exception as e:
        logger.error(f"Error creating database table 'sorting_rules': {e}")

    # --- MODIFIED: Create the ocr_results database table if it doesn't exist (add predicted_destination column) ---
    # This block should handle creating the table *with* the new column if it doesn't exist.
    # If the table already exists *without* the column, you need to run the ALTER TABLE statement manually (Step 1).
    try:
        conn = connect_to_db()
        if conn:
            cur = conn.cursor()
            # Check if the table exists and the column exists
            cur.execute("""
                SELECT EXISTS (
                   SELECT 1
                   FROM information_schema.tables
                   WHERE table_schema = 'public' -- Or your schema
                   AND table_name = 'ocr_results'
                );
            """)
            table_exists = cur.fetchone()[0]

            if not table_exists:
                 logger.info("Table 'ocr_results' does not exist. Creating it.")
                 cur.execute("""
                    CREATE TABLE ocr_results (
                        id SERIAL PRIMARY KEY,
                        text_content TEXT,
                        image_path VARCHAR(255), -- Store relative path or filename
                        timestamp TIMESTAMP,
                        predicted_destination VARCHAR(255) -- Added new column
                    )
                 """)
                 conn.commit()
                 logger.info("Database table 'ocr_results' created successfully with predicted_destination column.")
            else:
                 # Table exists, check if predicted_destination column exists
                 cur.execute("""
                     SELECT EXISTS (
                         SELECT 1
                         FROM information_schema.columns
                         WHERE table_schema = 'public' -- Or your schema
                         AND table_name = 'ocr_results'
                         AND column_name = 'predicted_destination'
                     );
                 """)
                 column_exists = cur.fetchone()[0]

                 if not column_exists:
                     logger.warning("Table 'ocr_results' exists but 'predicted_destination' column is missing. Attempting to add it.")
                     try:
                         cur.execute("ALTER TABLE ocr_results ADD COLUMN predicted_destination VARCHAR(255);")
                         conn.commit()
                         logger.info("Added 'predicted_destination' column to 'ocr_results' table.")
                     except Exception as alter_err:
                         logger.error(f"Failed to add 'predicted_destination' column: {alter_err}. You may need to add it manually.")
                         conn.rollback() # Rollback if ALTER fails
                 else:
                     logger.info("Database table 'ocr_results' and 'predicted_destination' column already exist.")


            cur.close()
            conn.close()
        else:
            logger.warning("Couldn't connect to database to check/create 'ocr_results' table")
    except Exception as e:
        logger.error(f"Error checking/creating database table 'ocr_results': {e}")


    # Create the template file if it doesn't exist (existing logic)
    create_index_template()

    # --- NEW: Create the rules list/add/edit templates if they don't exist ---
    # (You might prefer to create these manually as shown in the previous steps)
    # This is just a placeholder to ensure the directories exist if they don't
    os.makedirs('templates/rules', exist_ok=True)


    # Start the Flask server
    logger.info("Starting Flask server")
    # In a production environment, use a WSGI server like Gunicorn or uWSGI
    # For development, debug=True is fine.
    # host='0.0.0.0' makes it accessible externally (e.g., from image_watcher on another machine)
    app.run(debug=True, host='0.0.0.0', port=5000)
