import os
import time
import logging
import cv2
import pytesseract
import requests
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import shutil
from datetime import datetime
import numpy as np
import math # Import math for deskewing calculations
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# --- NEW: Import the servo controller module ---
import servo_controller

# --- NEW: Production mode flag ---
PRODUCTION_MODE = False  # Set to True in production to disable debug features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='logs/image_watcher.log'
)
logger = logging.getLogger('image_watcher')

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

# Configuration
WATCH_DIRECTORY = 'simulation/incoming_images'
PROCESSED_DIRECTORY = 'simulation/processed_images'
DEBUG_PREPROCESSING = not PRODUCTION_MODE  # Only save debug images in non-production mode

# Ensure your Flask server URL is correct
FLASK_SERVER_URL = 'http://localhost:5000/ocr_images.html' # Or your specific IP
DEBUG_PREPROCESSING = True  # Set to True to save debug images of preprocessing steps

MAX_IMAGE_WIDTH = 1200  # Resize larger images for faster processing
DESKEW_THRESHOLD = 2.0  # Only apply deskewing if skew angle is greater than this value (degrees)

# --- NEW: Global flag for servo system status ---
servos_ready = False

# Create directories if they don't exist
os.makedirs(WATCH_DIRECTORY, exist_ok=True)
os.makedirs(PROCESSED_DIRECTORY, exist_ok=True)
os.makedirs('logs', exist_ok=True)
if DEBUG_PREPROCESSING:
    os.makedirs('debug_images', exist_ok=True)

# --- NEW: Function to attempt deskewing ---
def deskew(image):
    """Attempts to deskew an image using moments."""
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Thresholding is often needed for moments to work well
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(thresh > 0))

    # Not enough points for reliable angle detection
    if len(coords) <= 20:  
        return 0.0
        
    # Calculate the angle of the minimum area bounding rectangle
    angle = cv2.minAreaRect(coords)[-1]

    # The angle returned by minAreaRect is in the range [-90, 0)
    # Adjust the angle to be in the range [-45, 45] or [0, 180] depending on convention
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle # Correct angle direction

    # Get the center of the image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Rotate the image
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    logger.debug(f"Deskewed image by {angle:.2f} degrees")
    return rotated, angle

# --- NEW: Function to resize image for faster processing ---
def resize_for_processing(image):
    """Resize image to a reasonable size for faster processing."""
    h, w = image.shape[:2]
    if w > MAX_IMAGE_WIDTH:
        ratio = MAX_IMAGE_WIDTH / w
        new_h = int(h * ratio)
        return cv2.resize(image, (MAX_IMAGE_WIDTH, new_h), interpolation=cv2.INTER_AREA)
    return image

def apply_advanced_preprocessing(img, debug_basename=None):
    """Apply advanced preprocessing techniques to improve OCR accuracy"""

    # Store the original for comparison
    original = img.copy()

    # --- NEW: First resize large images for faster processing ---
    img = resize_for_processing(img)
    if DEBUG_PREPROCESSING and debug_basename:
        cv2.imwrite(f"debug_images/{debug_basename}_0_resized.jpg", img)

    # --- NEW: Attempt Deskewing First ---
    deskewed_img, angle = deskew(original)
    if DEBUG_PREPROCESSING and debug_basename:
        cv2.imwrite(f"debug_images/{debug_basename}_0_deskewed.jpg", deskewed_img)
    img = deskewed_img # Use the deskewed image for subsequent steps

    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if DEBUG_PREPROCESSING and debug_basename:
        cv2.imwrite(f"debug_images/{debug_basename}_1_gray.jpg", gray)

    # Step 2: Apply bilateral filter to reduce noise while preserving edges
    # Tunable parameters: d (diameter), sigmaColor, sigmaSpace
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75) # Example parameters
    if DEBUG_PREPROCESSING and debug_basename:
        cv2.imwrite(f"debug_images/{debug_basename}_2_bilateral.jpg", bilateral)

    # Step 3: Apply adaptive thresholding to handle different lighting conditions
    # Tunable parameters: blockSize, C
    adaptive_thresh = cv2.adaptiveThreshold(
        bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2 # Example parameters
    )
    if DEBUG_PREPROCESSING and debug_basename:
        cv2.imwrite(f"debug_images/{debug_basename}_3_adaptive_thresh.jpg", adaptive_thresh)

    # Step 4: Apply morphological operations to clean up noise and improve text connectivity
    # Create a kernel for morphological operations
    # Tunable parameter: kernel size (e.g., (1,1), (2,2))
    kernel = np.ones((1, 1), np.uint8) # Example kernel size
    # Apply opening to remove small noise

    # This can be determined by analyzing the histogram of the thresholded image
    hist = cv2.calcHist([adaptive_thresh], [0], None, [2], [0, 256])
    noise_ratio = hist[0][0] / (hist[0][0] + hist[1][0])  # Ratio of black pixels
    
    if noise_ratio > 0.1:  # If there's significant noise
        opening = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
        if DEBUG_PREPROCESSING and debug_basename:
            cv2.imwrite(f"debug_images/{debug_basename}_4_opening.jpg", opening)
        
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        if DEBUG_PREPROCESSING and debug_basename:
            cv2.imwrite(f"debug_images/{debug_basename}_5_closing.jpg", closing)
    else:
        # Skip morphological operations if minimal noise
        closing = adaptive_thresh
        logger.debug("Skipping morphological operations due to low noise")


    opening = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
    if DEBUG_PREPROCESSING and debug_basename:
        cv2.imwrite(f"debug_images/{debug_basename}_4_opening.jpg", opening)

    # Step 5: Apply closing to fill small gaps in text
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    if DEBUG_PREPROCESSING and debug_basename:
        cv2.imwrite(f"debug_images/{debug_basename}_5_closing.jpg", closing)

    # Step 6: Apply unsharp masking to enhance edges
    # Blur the original grayscale with a Gaussian kernel
    # Tunable parameters: kernel size for blur, alpha, beta
    blurred = cv2.GaussianBlur(gray, (0, 0), 3) # Example kernel size
    unsharp_mask = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0) # Example alpha, beta
    if DEBUG_PREPROCESSING and debug_basename:
        cv2.imwrite(f"debug_images/{debug_basename}_6_unsharp_mask.jpg", unsharp_mask)

    # Step 7: Apply Otsu's thresholding on the unsharp masked image
    _, otsu_thresh = cv2.threshold(unsharp_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if DEBUG_PREPROCESSING and debug_basename:
        cv2.imwrite(f"debug_images/{debug_basename}_7_otsu_thresh.jpg", otsu_thresh)

    # Return multiple processed images for different OCR attempts
    return {
        'original': original, # Sometimes raw image works well
        'gray': gray,
        'adaptive_thresh': adaptive_thresh,
        'otsu_thresh': otsu_thresh,
        'closing': closing, # The result after morphological operations
        'unsharp_mask': unsharp_mask # The sharpened image
    }

def extract_text_from_image(image_path, debug_basename):
    """Apply OCR to extract text using multiple configurations and combine results."""
    img = cv2.imread(image_path)
    try:
        if img is None:
            logger.error(f"Failed to read image: {image_path}")
            return None

        # --- NEW: Check if image is too dark or too bright ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        avg_intensity = np.mean(gray)
        logger.debug(f"Average image intensity: {avg_intensity}")

        if avg_intensity < 30:  # Very dark image
            logger.info("Image is very dark, applying brightness correction")
            # Increase brightness
            img = cv2.convertScaleAbs(img, alpha=1.5, beta=30)
            if DEBUG_PREPROCESSING and debug_basename:
                cv2.imwrite(f"debug_images/{debug_basename}_brightness_corrected.jpg", img)
        elif avg_intensity > 220:  # Very bright image
            logger.info("Image is very bright, applying contrast enhancement")
            # Increase contrast
            img = cv2.convertScaleAbs(img, alpha=1.2, beta=-20)
            if DEBUG_PREPROCESSING and debug_basename:
                cv2.imwrite(f"debug_images/{debug_basename}_contrast_enhanced.jpg", img)
            
            # Get multiple enhanced images from preprocessing
            processed_images = apply_advanced_preprocessing(img, debug_basename)

    except Exception as e:  # <-- REQUIRED EXCEPT BLOCK
        logger.error(f"Image processing error: {str(e)}")
        return None

        
    # --- NEW: Define Multiple Tesseract Configurations ---
    # Experiment with these configurations based on your image characteristics
    ocr_configs = {
        'psm4_oem3_adaptive': {'image': processed_images['adaptive_thresh'], 'config': "--psm 4 --oem 3"}, # Single column, LSTM
        'psm6_oem3_otsu': {'image': processed_images['otsu_thresh'], 'config': "--psm 6 --oem 3"},     # Uniform block, LSTM
        'psm11_oem3_original': {'image': processed_images['original'], 'config': "--psm 11 --oem 3"}, # Sparse text, LSTM
        'psm6_oem1_closing': {'image': processed_images['closing'], 'config': "--psm 6 --oem 1"},     # Uniform block, LSTM+Legacy
        'psm3_oem3_unsharp': {'image': processed_images['unsharp_mask'], 'config': "--psm 3 --oem 3"}, # Auto page seg, LSTM
        'psm6_oem0_gray': {'image': processed_images['gray'], 'config': "--psm 6 --oem 0"},           # Uniform block, Legacy only
    }

    # --- NEW: Optional: Character Whitelisting/Blacklisting ---
    # Uncomment and modify if you know the character set is limited
    # common_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,#-"
    # for key in ocr_configs:
    #     ocr_configs[key]['config'] += f" -c tessedit_char_whitelist='{common_chars}'"

    all_results = {}
    best_text = ""
    best_len = 0    

    for name, params in ocr_configs.items():
            try:
                if params['image'] is not None:
                    text = pytesseract.image_to_string(params['image'], config=params['config'])
                    all_results[name] = text.strip()
                    logger.info(f"OCR Result ({name}): {text.strip()[:100]}...")
                    
                    # Check if we have a good result already (more than 50 chars)
                    if len(text.strip()) > 50:
                        logger.info(f"Found good result with config {name}, skipping additional OCR attempts")
                        best_text = text.strip()
                        best_len = len(best_text)
                        break
                else:
                    logger.warning(f"Skipping OCR config '{name}' due to missing processed image.")
                    all_results[name] = ""
            except Exception as e:
                logger.error(f"Error running OCR with config '{name}': {e}")
                all_results[name] = ""
                

         # --- NEW: Only add additional configs if we don't have a good result yet ---
    if best_len <= 50:
            logger.info("Basic OCR configs gave insufficient results, trying additional configs")
            
            # Add more complex configurations
            additional_configs = {
                'psm11_oem3_original': {'image': processed_images['original'], 'config': "--psm 11 --oem 3"},
                'psm6_oem1_closing': {'image': processed_images['closing'], 'config': "--psm 6 --oem 1"},
            }
       
            # Add these to our ocr_configs dictionary
            ocr_configs.update(additional_configs)

            # Process these additional configs
            for name, params in additional_configs.items():
                try:
                    if params['image'] is not None:
                        text = pytesseract.image_to_string(params['image'], config=params['config'])
                        all_results[name] = text.strip()
                        logger.info(f"OCR Result ({name}): {text.strip()[:100]}...")
                    else:
                        logger.warning(f"Skipping OCR config '{name}' due to missing processed image.")
                        all_results[name] = ""
                except Exception as e:
                    logger.error(f"Error running OCR with config '{name}': {e}")
                    all_results[name] = ""

        # --- Apply scoring to select the best result ---
    if not best_text:  # Only if we haven't already found a good result
            target_keywords = ["Tungkong Mangga", "Mulawin", "Kaypian", "Poblacion", "Highway", "Street", "Brgy", "ZIP Code"]
            best_score = -1

    # --- NEW: More Intelligent Result Combination ---

    # Simple scoring: prioritize results that contain more target keywords,
    # then by length.
    scored_results = []
    for name, text in all_results.items():
        keyword_count = sum(keyword.lower() in text.lower() for keyword in target_keywords)
        # Combine keyword count and length for a score
        score = keyword_count * 1000 + len(text) # Keywords give a higher weight
        scored_results.append({'name': name, 'text': text, 'score': score, 'keyword_count': keyword_count})
        logger.debug(f"Config '{name}' score: {score} (Keywords: {keyword_count}, Length: {len(text)})")

    # Sort results by score in descending order
    scored_results.sort(key=lambda x: x['score'], reverse=True)

    if scored_results:
        best_result_info = scored_results[0]
        best_text = best_result_info['text']
        logger.info(f"Best OCR result chosen from '{best_result_info['name']}' with score {best_result_info['score']} (Keywords: {best_result_info['keyword_count']}).")
    else:
        logger.warning("No OCR results were successfully obtained.")
        best_text = "" # Ensure best_text is empty if no results

    # Log the length of each result for debugging
    # logger.info(f"OCR Result lengths: {', '.join([f'{k}: {len(v)}' for k, v in all_results.items()])}")

    # --- NEW: Memory cleanup ---
    processed_images.clear()
    return best_text.strip()

def process_image(image_path):
    """Process an image with OCR, trigger servos, and send it to the Flask server"""
    # --- Use the global flag ---
    global servos_ready
    try:
        logger.info(f"Processing image: {image_path}")

        # Make sure the file exists and is readable
        if not os.path.isfile(image_path):
            logger.error(f"File not found: {image_path}")
            return False

        # Get the basename for debug images
        basename = os.path.basename(image_path).split('.')[0]

        # Extract text from the image using the improved function
        text = extract_text_from_image(image_path, basename) # <-- Uses the improved OCR

        if not text:
            logger.warning(f"No text extracted from {image_path}")
            text = "NO TEXT DETECTED" # Assign default text for logging/potential server upload
            # --- NEW: No text means no servo action needed ---
            logger.info("No text detected, skipping servo action.")
            predicted_destination = "NO TEXT DETECTED" # Set a default destination for no text
            match_score = 0 # Default score

        else:
            logger.info(f"Extracted text length: {len(text)}")
            logger.info(f"First 100 chars: {text[:100]}...")

            # --- NEW: Send Extracted Text to Flask Server for Sorting Lookup ---
            # We are now sending the extracted text to the Flask server
            # The Flask server will perform the sorting rule lookup and return the destination
            predicted_destination = "LOOKUP FAILED" # Default if server lookup fails
            match_score = 0 # Default score

            try:
                # Prepare data to send to Flask server
                # Send the image file and the extracted text
                with open(image_path, 'rb') as img_file:
                     files = {'image': (os.path.basename(image_path), img_file, 'image/jpeg')}
                     # Send extracted text as form data
                     data = {'extracted_text': text}

                     logger.info(f"Sending image and text to Flask server: {FLASK_SERVER_URL}")
                     response = requests.post(FLASK_SERVER_URL, files=files, data=data)

                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Server response: {result}")

                    # --- NEW: Get predicted_destination and match_score from Flask server response ---
                    predicted_destination = result.get('predicted_destination', 'UNKNOWN')
                    match_score = result.get('match_score', 0)
                    logger.info(f"Received predicted_destination from server: {predicted_destination} (Score: {match_score})")

                    # --- NEW: Trigger Servo Based on Predicted Destination from Server ---
                    if servos_ready:
                        logger.info(f"Attempting to trigger servo for destination: {predicted_destination}")
                        # You need to modify servo_controller.sort_parcel to accept the destination string
                        # and map it to a servo action.
                        servo_controller.sort_parcel_by_destination(predicted_destination) # Assuming a new function name
                    else:
                        logger.warning("Servo system not ready, skipping sorting action.")
                        logger.info(f"(Simulation) Would sort to: {predicted_destination}")


                    # Move to processed directory (do this AFTER all processing is done)
                    processed_path = os.path.join(PROCESSED_DIRECTORY, os.path.basename(image_path))
                    try:
                        shutil.move(image_path, processed_path)
                        logger.info(f"Image moved to processed directory: {processed_path}")
                        return True # Indicate processing completed successfully
                    except Exception as move_err:
                        logger.error(f"Failed to move image {image_path} to {processed_path}: {move_err}")
                        # Decide if this is a fatal error for the processing step
                        return False # Indicate processing didn't fully complete

                else:
                    logger.error(f"Server error processing image: {response.status_code} - {response.text}")
                    return False # Indicate processing didn't fully complete
            except requests.RequestException as e:
                logger.error(f"Request error when sending to server: {str(e)}")
                return False # Indicate processing didn't fully complete

        # --- Existing logic: Move to processed directory (Removed from original location) ---
        # The move logic is now inside the successful server response block.

    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}", exc_info=True) # Add exc_info for traceback
        return False # Indicate processing failed

class ImageHandler(FileSystemEventHandler):
    def on_created(self, event):
        """Handle new image file creation event"""
        if event.is_directory:
            return

        # Check if it's an image file
        if event.src_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            logger.info(f"New image detected: {event.src_path}")

            # Give the file system a moment to finish writing the file
            # This helps prevent reading incomplete files
            file_ready = False
            attempts = 0
            max_attempts = 10 # Increased attempts
            initial_size = -1
            # Wait for file size to stabilize
            while not file_ready and attempts < max_attempts:
                try:
                    current_size = os.path.getsize(event.src_path)
                    # Check if size is greater than 0 and hasn't changed in the last check
                    if current_size > 0 and current_size == initial_size:
                         file_ready = True
                         logger.info(f"File size stable, proceeding with processing: {event.src_path} (Size: {current_size})")
                    else:
                         initial_size = current_size
                         logger.debug(f"Waiting for file write to complete for {event.src_path}. Size: {current_size}. Attempt: {attempts+1}")
                         time.sleep(0.5) # Wait a bit longer
                except OSError as e:
                     logger.warning(f"Error accessing file {event.src_path}, possibly still writing: {e}")
                     time.sleep(0.5) # Wait before retrying access
                attempts += 1

            if file_ready:
                 # Process the image
                 process_image(event.src_path)
            else:
                 logger.error(f"File {event.src_path} did not become ready after {max_attempts} attempts. Skipping.")


def extract_tracking_info(text):
    tracking_number = None
    recipient = None
    date = None
    address = None
    
    # Example patterns (you would use regex or more sophisticated NLP)
    if "TRACKING" in text:
        # Try to find tracking number pattern
        # e.g., tracking_number = re.search(r'TRACKING[:#\s]+([A-Z0-9]+)', text)
        pass
        
    if "RECIPIENT" in text or "TO:" in text:
        # Try to extract recipient name
        pass
        
    # Extract date and address similarly
    
    return {
        'tracking_number': tracking_number,
        'recipient': recipient,
        'date': date,
        'address': address
    }

def scan_existing_images():
    """Scan and process any existing images in the watch directory"""
    global servos_ready # Ensure we use the global flag
    logger.info(f"Checking for existing images in {WATCH_DIRECTORY}")
    processed_count = 0

    # Get list of files first to avoid issues with files being moved during iteration
    files_to_process = [f for f in os.listdir(WATCH_DIRECTORY) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for filename in files_to_process:
        file_path = os.path.join(WATCH_DIRECTORY, filename)
        logger.info(f"Found existing image: {file_path}")
        # Ensure file is ready before processing (simple check for existing files)
        try:
            # Basic check: Wait a tiny bit after finding it
            time.sleep(0.2)
            if os.path.exists(file_path): # Re-check existence
                 if process_image(file_path):
                      processed_count += 1
            else:
                 logger.warning(f"Existing file disappeared before processing: {file_path}")
        except Exception as e:
             logger.error(f"Error processing existing file {file_path}: {e}")

    logger.info(f"Processed {processed_count} existing images")
    return processed_count

def main():
    global servos_ready # Declare intention to modify the global variable

    # --- NEW: Initialize Servos ---
    logger.info("--- Initializing Servo System ---")
    servos_ready = servo_controller.setup_servos()
    if not servos_ready:
        logger.warning("!!! Servo system failed to initialize. Running without servo control. !!!")
        # Decide if you want to exit if servos are critical
        # exit()
    logger.info("--- Servo Initialization Complete ---")

    # Process any existing images first
    scan_existing_images()

    # Set up the file system watcher
    logger.info(f"Setting up file system watcher for {WATCH_DIRECTORY}")
    event_handler = ImageHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_DIRECTORY, recursive=False)
    observer.start()

    try:
        logger.info("Image watcher started. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Image watcher stopping...")
        observer.stop()
        logger.info("Observer stopped.")

    # --- NEW: Release Servos on Exit ---
    finally: # Use finally to ensure this runs even if errors occur
        if servos_ready:
            logger.info("Releasing servos...")
            servo_controller.release_servos()
            logger.info("Servos released.")
        else:
            logger.info("Servo system was not ready, no servos to release.")

        # Wait for the observer thread to finish
        observer.join()
        logger.info("Image watcher shutdown complete.")


if __name__ == "__main__":
    # --- NEW: Ensure servo_controller.py is in the same directory ---
    # --- or accessible in the Python path ---

    # --- Make sure Tesseract path is set if needed ---
    # Example: Set if not in system PATH (uncomment and adjust if needed)
    # try:
    #    # Use r'' for raw string, and forward slashes generally work on Windows
    #    pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
    #    pytesseract.get_tesseract_version() # Check if path is valid
    # except Exception as e:
    #    logger.error(f"Tesseract not found or path incorrect: {e}")
    #    logger.error("Please ensure Tesseract is installed and the path is correct.")
    #    # Decide if you want to exit if Tesseract is critical
    #    # exit()

    main()
