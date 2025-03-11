from PIL import Image
import os

def create_favicon(input_path, output_path, size=(32, 32)):
    # Open the image
    with Image.open(input_path) as img:
        # Convert to RGBA if not already
        img = img.convert('RGBA')
        
        # Create a new image with white background
        background = Image.new('RGBA', img.size, (255, 255, 255, 255))
        
        # Paste the image on the background using alpha channel
        background.paste(img, (0, 0), img)
        
        # Resize the image
        img_resized = background.resize(size, Image.Resampling.LANCZOS)
        
        # Save as ICO
        img_resized.save(output_path, format='ICO', sizes=[(32, 32)])

if __name__ == '__main__':
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Input and output paths
    input_path = os.path.join(script_dir, 'CellDISECT_Logo_whitebg.png')
    output_path = os.path.join(script_dir, 'favicon.ico')
    
    create_favicon(input_path, output_path) 