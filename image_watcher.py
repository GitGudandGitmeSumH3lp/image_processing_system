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

# --- NEW: Import the servo controller module ---
import servo_controller

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
FLASK_SERVER_URL = 'http://localhost:5000/upload_image'
DEBUG_PREPROCESSING = True  # Set to True to save debug images of preprocessing steps

# --- NEW: Global flag for servo system status ---
servos_ready = False

# Create directories if they don't exist
os.makedirs(WATCH_DIRECTORY, exist_ok=True)
os.makedirs(PROCESSED_DIRECTORY, exist_ok=True)
os.makedirs('logs', exist_ok=True)
if DEBUG_PREPROCESSING:
    os.makedirs('debug_images', exist_ok=True)

def apply_advanced_preprocessing(img, debug_basename=None):
    """Apply advanced preprocessing techniques to improve OCR accuracy"""
    
    # Store the original for comparison
    original = img.copy()
    
    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if DEBUG_PREPROCESSING and debug_basename:
        cv2.imwrite(f"debug_images/{debug_basename}_1_gray.jpg", gray)
    
    # Step 2: Apply bilateral filter to reduce noise while preserving edges
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    if DEBUG_PREPROCESSING and debug_basename:
        cv2.imwrite(f"debug_images/{debug_basename}_2_bilateral.jpg", bilateral)
    
    # Step 3: Apply adaptive thresholding to handle different lighting conditions
    adaptive_thresh = cv2.adaptiveThreshold(
        bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    if DEBUG_PREPROCESSING and debug_basename:
        cv2.imwrite(f"debug_images/{debug_basename}_3_adaptive_thresh.jpg", adaptive_thresh)
    
    # Step 4: Apply morphological operations to clean up noise and improve text connectivity
    # Create a kernel for morphological operations
    kernel = np.ones((1, 1), np.uint8)
    # Apply opening to remove small noise
    opening = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
    if DEBUG_PREPROCESSING and debug_basename:
        cv2.imwrite(f"debug_images/{debug_basename}_4_opening.jpg", opening)
    
    # Step 5: Apply closing to fill small gaps in text
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    if DEBUG_PREPROCESSING and debug_basename:
        cv2.imwrite(f"debug_images/{debug_basename}_5_closing.jpg", closing)
    
    # Step 6: Apply unsharp masking to enhance edges
    # Blur the original grayscale with a Gaussian kernel
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    unsharp_mask = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    if DEBUG_PREPROCESSING and debug_basename:
        cv2.imwrite(f"debug_images/{debug_basename}_6_unsharp_mask.jpg", unsharp_mask)
    
    # Step 7: Apply Otsu's thresholding on the unsharp masked image
    _, otsu_thresh = cv2.threshold(unsharp_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if DEBUG_PREPROCESSING and debug_basename:
        cv2.imwrite(f"debug_images/{debug_basename}_7_otsu_thresh.jpg", otsu_thresh)
    
    # Return both the adaptive thresholding result and Otsu's result
    # We'll use both for OCR and combine results
    return adaptive_thresh, otsu_thresh

def extract_text_from_image(image_path, debug_basename):
    """Apply OCR to extract text with improved techniques"""
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Failed to read image: {image_path}")
        return None
    
    # Get enhanced images
    adaptive_thresh, otsu_thresh = apply_advanced_preprocessing(img, debug_basename)
    
    # Apply OCR with different configurations and combine results
    results = []
    
    # Configuration 1: Basic config on adaptive threshold
    config1 = "--psm 4"  # Assume a single column of text
    text1 = pytesseract.image_to_string(adaptive_thresh, config=config1)
    results.append(text1)
    
    # Configuration 2: Different PSM on Otsu threshold
    config2 = "--psm 6"  # Assume a uniform block of text
    text2 = pytesseract.image_to_string(otsu_thresh, config=config2)
    results.append(text2)
    
    # Configuration 3: Original image with specialized config for forms
    config3 = "--psm 11 --oem 3"  # Sparse text with OEM 3 (LSTM only)
    text3 = pytesseract.image_to_string(img, config=config3)
    results.append(text3)
    
    # Choose the best result (we'll take the longest as it likely has the most information)
    results.sort(key=len, reverse=True)
    best_text = results[0]
    
    # Log the length of each result for debugging
    logger.info(f"OCR Result lengths - Adaptive: {len(text1)}, Otsu: {len(text2)}, Original: {len(text3)}")
    
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
        
        # Extract text from the image
        text = extract_text_from_image(image_path, basename)
        
        if not text:
            logger.warning(f"No text extracted from {image_path}")
            text = "NO TEXT DETECTED" # Assign default text for logging/potential server upload
            # --- NEW: No text means no servo action needed ---
            logger.info("No text detected, skipping servo action.")
        else:
            logger.info(f"Extracted text length: {len(text)}")
            logger.info(f"First 100 chars: {text[:100]}...")
            
            # --- NEW: Trigger Servo Based on OCR Text ---
            if servos_ready:
                logger.info("Attempting to trigger servo based on OCR text...")
                servo_controller.sort_parcel(text) # Pass extracted text
            else:
                logger.warning("Servo system not ready, skipping sorting action.")
                # You could add logging here to simulate what would happen
                # address_found = False
                # for key_addr in servo_controller.ADDRESS_TO_SERVO_CHANNEL:
                #    if key_addr.lower() in text.lower():
                #        logger.info(f"(Simulation) Would sort to: {key_addr}")
                #        address_found = True
                #        break
                # if not address_found:
                #    logger.info("(Simulation) No target address found.")


        # --- Existing logic: Send to Flask server ---
        try:
            with open(image_path, 'rb') as img_file:
                files = {'image': (os.path.basename(image_path), img_file, 'image/jpeg')}
                # You might want to include the extracted text in the data sent to Flask
                # data = {'extracted_text': text} 
                # response = requests.post(FLASK_SERVER_URL, files=files, data=data)
                response = requests.post(FLASK_SERVER_URL, files=files) 
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Image sent to server successfully. Result ID: {result.get('result_id', 'unknown')}")
                
                # Move to processed directory (do this AFTER all processing is done)
                processed_path = os.path.join(PROCESSED_DIRECTORY, os.path.basename(image_path))
                try:
                    shutil.move(image_path, processed_path)
                    logger.info(f"Image moved to processed directory: {processed_path}")
                    return True
                except Exception as move_err:
                    logger.error(f"Failed to move image {image_path} to {processed_path}: {move_err}")
                    # Decide if this is a fatal error for the processing step
                    return False # Indicate processing didn't fully complete

            else:
                logger.error(f"Server error: {response.status_code} - {response.text}")
                return False # Indicate processing didn't fully complete
        except requests.RequestException as e:
            logger.error(f"Request error when sending to server: {str(e)}")
            return False # Indicate processing didn't fully complete
            
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}", exc_info=True) # Add exc_info for traceback
        return False

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
            max_attempts = 5
            initial_size = -1
            while not file_ready and attempts < max_attempts:
                try:
                    current_size = os.path.getsize(event.src_path)
                    if current_size == initial_size and current_size > 0:
                         file_ready = True
                         logger.info(f"File size stable, proceeding with processing: {event.src_path}")
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


def scan_existing_images():
    """Scan and process any existing images in the watch directory"""
    global servos_ready # Ensure we use the global flag
    logger.info(f"Checking for existing images in {WATCH_DIRECTORY}")
    processed_count = 0
    
    for filename in os.listdir(WATCH_DIRECTORY):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
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
 pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe' # Example for Linux
 pytesseract.get_tesseract_version() # Check if path is valid
    # except Exception as e:
#logger.error(f"Tesseract not found or path incorrect: {e}")
#logger.error("Please ensure Tesseract is installed and the path is correct.")
    #    # Decide if you want to exit if Tesseract is critical
    #    # exit() 

main()