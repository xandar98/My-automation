
from flask import Flask, render_template, request, jsonify, send_file
import requests
import os
from io import BytesIO
import zipfile
from datetime import datetime
import base64

app = Flask(__name__)

# Hugging Face API configuration - using Stable Diffusion XL
HF_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
HF_HEADERS = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}

def query_huggingface(prompt):
    payload = {"inputs": prompt}
    
    try:
        response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=60)
        
        if response.status_code == 200:
            return response.content
        elif response.status_code == 503:
            print(f"Model loading, status: {response.status_code}")
            return None
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_images():
    data = request.get_json()
    prompt = data.get('prompt')
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    api_key = os.getenv('HUGGINGFACE_API_KEY')
    if not api_key:
        return jsonify({'error': 'Hugging Face API key not configured'}), 500
    
    print(f"Using API key: {api_key[:10]}...") # Debug print (first 10 chars)
    
    images = []
    errors = []
    
    for i in range(4):
        # Add variation to each image
        varied_prompt = f"{prompt}, high quality, detailed, 4k resolution"
        print(f"Generating image {i+1} with prompt: {varied_prompt}")
        
        image_data = query_huggingface(varied_prompt)
        
        if image_data:
            try:
                # Convert to base64 for frontend display
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                images.append({
                    'id': i,
                    'data': image_base64
                })
                print(f"Successfully generated image {i+1}")
            except Exception as e:
                error_msg = f"Error encoding image {i+1}: {str(e)}"
                print(error_msg)
                errors.append(error_msg)
        else:
            error_msg = f"Failed to generate image {i+1}"
            print(error_msg)
            errors.append(error_msg)
    
    if not images:
        return jsonify({'error': f'Failed to generate any images. Errors: {"; ".join(errors)}'}), 500
    
    response_data = {'images': images}
    if errors:
        response_data['warnings'] = errors
    
    return jsonify(response_data)

@app.route('/download/<int:image_id>')
def download_image(image_id):
    # This would need to store images temporarily or use session storage
    # For simplicity, we'll regenerate or use a different approach
    return jsonify({'error': 'Direct download not implemented in this version'}), 501

@app.route('/download_all', methods=['POST'])
def download_all():
    data = request.get_json()
    images_data = data.get('images', [])
    
    if not images_data:
        return jsonify({'error': 'No images to download'}), 400
    
    # Create a zip file in memory
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, img_data in enumerate(images_data):
            image_bytes = base64.b64decode(img_data)
            filename = f"generated_image_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            zip_file.writestr(filename, image_bytes)
    
    zip_buffer.seek(0)
    
    return send_file(
        zip_buffer,
        as_attachment=True,
        download_name=f"generated_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
        mimetype='application/zip'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
