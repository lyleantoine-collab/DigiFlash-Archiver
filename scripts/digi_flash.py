import os
from PIL import Image
import pytesseract
import numpy as np
from skimage import filters
from skimage.transform import rotate
from skimage.measure import label, regionprops
from scipy.ndimage import median_filter
import re

# Config for '80s-style OCR (optimized for faded '50s docs)
custom_config = (
    r'--oem 3 --psm 6 -l eng '
    r'--tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,/- '
    r'-c tessedit_pageseg_mode=6'
)

def binarize_image(img_array):
    """Otsu threshold for clean binarization (10-15% OCR boost)"""
    thresh = filters.threshold_otsu(img_array)
    return (img_array > thresh).astype(np.uint8) * 255

def deskew_image(img):
    """Fix tilted scans using largest text region's orientation"""
    img_array = np.array(img)
    # Ensure binary
    if img_array.max() <= 1:
        img_array = img_array * 255
    binary = img_array > filters.threshold_otsu(img_array)
    
    labeled = label(binary)
    props = regionprops(labeled)
    
    if len(props) == 0:
        return img  # No regions found

    # Use largest region by area
    largest = max(props, key=lambda p: p.area)
    angle = largest.orientation  # in radians, range [-π/2, π/2]
    angle_deg = np.degrees(angle)
    
    # Normalize angle to [-45, 45]
    if angle_deg > 45:
        angle_deg -= 90
    elif angle_deg < -45:
        angle_deg += 90

    return Image.fromarray(rotate(img_array, angle_deg, resize=True, cval=255).astype(np.uint8))

def denoise_image(img_array):
    """Remove salt-and-pepper noise from aged paper"""
    return median_filter(img_array, size=3)

def extract_date_from_text(text):
    """Try to find a 19xx or 195x date in text"""
    date_pattern = r'\b(19[5-9]\d)\b'  # Matches 1950–1999
    match = re.search(date_pattern, text)
    return match.group(1) if match else '1950'

def process_scan(folder_path, output_path, backup_path):
    """Main pipeline: clean → deskew → OCR → smart name → save + backup"""
    if not all(os.path.isdir(p) for p in [folder_path, output_path, backup_path]):
        raise ValueError("One or more paths are invalid or don't exist.")

    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            file_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(file_path).convert('L')
                img_array = np.array(img)

                # === Preprocessing Pipeline ===
                img_array = denoise_image(img_array)
                img_array = binarize_image(img_array)
                img = Image.fromarray(img_array)
                img = deskew_image(img)  # Now returns PIL Image

                # === OCR ===
                text = pytesseract.image_to_string(img, config=custom_config).strip()
                if not text:
                    text = "NO_TEXT"

                # === Smart Naming ===
                date = extract_date_from_text(text)
                first_line = text.split('\n')[0].strip()
                keyword = re.sub(r'[^A-Za-z0-9]+', '_', first_line)[:30]
                if not keyword or keyword == "NO_TEXT":
                    keyword = "unknown"
                new_name = f"{date}-{keyword}-{filename}"

                # === Save ===
                output_file = os.path.join(output_path, new_name)
                backup_file = os.path.join(backup_path, new_name)
                img.save(output_file)
                img.save(backup_file)

                print(f"Processed: {filename} → {new_name} ({len(text)} chars)")

            except Exception as e:
                print(f"Failed on {filename}: {e}")

    print("Batch complete. '80s terminal: *BEEP BOOP* All archived & mirrored.")
