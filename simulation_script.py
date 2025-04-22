import os
import time
import cv2
import numpy as np
from datetime import datetime
import random
import shutil
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='logs/simulator.log'
)
logger = logging.getLogger('esp32cam_simulator')

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

# Sample parcel data for Bulacan, Philippines near San Jose Del Monte
PARCEL_DATA = [
    {
        "recipient_name": "Maria Santos",
        "address": "123 Poblacion St., Brgy. Mulawin",
        "city": "San Jose Del Monte",
        "province": "Bulacan",
        "zip_code": "3023",
        "contact": "0917-123-4567",
        "tracking_number": "",
        "delivery_status": ""
    },
    {
        "recipient_name": "Juan Dela Cruz",
        "address": "45 Maharlika Highway, Brgy. Tungkong Mangga",
        "city": "San Jose Del Monte",
        "province": "Bulacan",
        "zip_code": "3023",
        "contact": "0918-765-4321",
        "tracking_number": "",
        "delivery_status": ""
    },
    {
        "recipient_name": "Elena Reyes",
        "address": "78 Quirino Highway, Brgy. Kaypian",
        "city": "San Jose Del Monte",
        "province": "Bulacan",
        "zip_code": "3023",
        "contact": "0919-876-5432",
        "tracking_number": "",
        "delivery_status": ""
    }
]

# Delivery statuses
DELIVERY_STATUSES = [
    "For Pickup",
    "In Transit",
    "Out for Delivery",
    "Delivered",
    "Failed Delivery Attempt",
    "Return to Sender"
]

def generate_tracking_number():
    """Generate a random tracking number"""
    prefix = random.choice(["PHL", "SJDM", "BUL"])
    numbers = ''.join(random.choices('0123456789', k=10))
    return f"{prefix}{numbers}"

def generate_parcel_image(output_path):
    """Generate an image with parcel details for OCR testing"""
    # Create a white image
    img = np.ones((600, 800, 3), np.uint8) * 255
    
    # Add title
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "PARCEL DELIVERY DETAILS", (200, 40), font, 1, (0, 0, 0), 2)
    
    # Get random parcel data
    parcel = random.choice(PARCEL_DATA).copy()
    
    # Add tracking number and status
    parcel["tracking_number"] = generate_tracking_number()
    parcel["delivery_status"] = random.choice(DELIVERY_STATUSES)
    
    # Current date and time
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Draw details on image
    details = [
        f"Tracking #: {parcel['tracking_number']}",
        f"Status: {parcel['delivery_status']}",
        f"Date: {timestamp}",
        "",
        f"Recipient: {parcel['recipient_name']}",
        f"Address: {parcel['address']}",
        f"City: {parcel['city']}",
        f"Province: {parcel['province']}",
        f"ZIP Code: {parcel['zip_code']}",
        f"Contact #: {parcel['contact']}"
    ]
    
    y_position = 100
    for detail in details:
        cv2.putText(img, detail, (50, y_position), font, 0.7, (0, 0, 0), 1)
        y_position += 40
    
    # Add a barcode-like element for visual effect
    y_pos = 450
    for i in range(30):
        width = random.randint(1, 5)
        x_pos = 100 + i * 20
        cv2.rectangle(img, (x_pos, y_pos), (x_pos + width, y_pos + 80), (0, 0, 0), -1)
    
    # Add company logo placeholder
    cv2.rectangle(img, (600, 100), (750, 200), (200, 200, 200), -1)
    cv2.putText(img, "LOGO", (650, 160), font, 1, (100, 100, 100), 2)
    
    # Apply a slight blur to simulate camera capture
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Add random noise to make OCR more challenging but realistic
    noise = np.zeros(img.shape, np.uint8)
    cv2.randn(noise, 0, 10)
    img = cv2.add(img, noise)
    
    # Save the image with JPEG compression
    compression_params = [int(cv2.IMWRITE_JPEG_QUALITY), 92]
    success = cv2.imwrite(output_path, img, compression_params)
    
    if not success:
        logger.error(f"Failed to create image: {output_path}")
        return None

    logger.info(f"Generated parcel detail image: {output_path}")
    return output_path

def copy_test_image(source_folder, destination):
    """Copy a test image from a source folder to the destination"""
    if not os.path.exists(source_folder) or not os.listdir(source_folder):
        logger.warning(f"Source folder {source_folder} doesn't exist or is empty")
        return None
    
    # Get a random image from the source folder
    image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        logger.warning(f"No image files found in {source_folder}")
        return None
    
    selected_image = random.choice(image_files)
    source_path = os.path.join(source_folder, selected_image)
    shutil.copy(source_path, destination)
    logger.info(f"Copied test image: {source_path} to {destination}")
    return destination

def main():
    parser = argparse.ArgumentParser(description='Parcel Detail Image Simulator')
    parser.add_argument('--mode', choices=['generate', 'copy'], default='generate',
                        help='Mode: generate new images or copy existing test images')
    parser.add_argument('--source', default='test_images',
                        help='Source folder for test images (for copy mode)')
    parser.add_argument('--interval', type=int, default=10,
                        help='Interval in seconds between images')
    parser.add_argument('--count', type=int, default=0,
                        help='Number of images to generate (0 for infinite)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = 'simulation/incoming_images'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Create test images directory if in copy mode
    if args.mode == 'copy':
        os.makedirs(args.source, exist_ok=True)
        if not os.listdir(args.source):
            logger.warning(f"Test images folder {args.source} is empty. Please add some test images.")
    
    logger.info(f"Starting parcel detail simulator in {args.mode} mode")
    logger.info(f"Images will be saved to {output_dir}")
    logger.info(f"Interval between images: {args.interval} seconds")
    
    count = 0
    try:
        while args.count == 0 or count < args.count:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"parcel_{timestamp}.jpg")
            
            if args.mode == 'generate':
                generate_parcel_image(output_path)
            else:  # copy mode
                if not copy_test_image(args.source, output_path):
                    logger.error("Failed to copy test image. Exiting.")
                    break
            
            count += 1
            if args.count > 0:
                logger.info(f"Generated {count}/{args.count} images")
            else:
                logger.info(f"Generated image #{count}")
            
            time.sleep(args.interval)
    
    except KeyboardInterrupt:
        logger.info("Simulator stopped by user")
    
    logger.info(f"Simulator finished after generating {count} images")

if __name__ == "__main__":
    main()