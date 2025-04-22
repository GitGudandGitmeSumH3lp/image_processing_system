import logging

logger = logging.getLogger('servo_controller')

# Configure logging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

# Common locations in SJDM Bulacan for sorting
ADDRESS_TO_SERVO_CHANNEL = {
    "Tungkong Mangga": 1,
    "Mulawin": 2,
    "Kaypian": 3,
    "Poblacion": 4
}

def setup_servos():
    """Set up servos - this is a stub for Windows development"""
    logger.info("STUB: Setting up servo motors (simulation mode)")
    return True  # Return True to indicate "servos" are ready

def sort_parcel(ocr_text):
    """Sort a parcel based on the OCR text - this is a stub for Windows development"""
    logger.info("STUB: Would sort parcel based on extracted text")
    
    address_found = False
    for key_addr in ADDRESS_TO_SERVO_CHANNEL:
        if key_addr.lower() in ocr_text.lower():
            logger.info(f"SIMULATION: Would sort to channel {ADDRESS_TO_SERVO_CHANNEL[key_addr]} for address: {key_addr}")
            address_found = True
            break
    
    if not address_found:
        logger.info("SIMULATION: No matching address found in OCR text")
    
    return address_found

def release_servos():
    """Release servo motors - this is a stub for Windows development"""
    logger.info("STUB: Releasing servo motors (simulation mode)")
    return True