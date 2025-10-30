import requests
import json
import os
from PIL import Image
import io

def test_dog_classifier_api():
    url = "http://127.0.0.1:8000/api/dog-classify/"
    
    test_image = Image.new('RGB', (224, 224), color='red')
    
    img_byte_arr = io.BytesIO()
    test_image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    files = {
        'image': ('test_image.jpg', img_byte_arr, 'image/jpeg')
    }
    
    data = {
        'species': 'dog'
    }
    
    try:
        print("Testing dog classifier API...")
        print(f"URL: {url}")
        
        response = requests.post(url, files=files, data=data)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("[SUCCESS] API Response:")
            print(json.dumps(result, indent=2))
            
            if 'predictions' in result and isinstance(result['predictions'], list):
                print(f"[SUCCESS] Found {len(result['predictions'])} predictions")
                for i, pred in enumerate(result['predictions']):
                    if 'breed' in pred and 'probability' in pred:
                        print(f"  {i+1}. {pred['breed']}: {pred['probability']}%")
                    else:
                        print(f"  [ERROR] Invalid prediction format: {pred}")
            else:
                print("[ERROR] Invalid response format - missing 'predictions' field")
        else:
            print(f"[ERROR] API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("[ERROR] Connection Error: Make sure Django server is running on http://127.0.0.1:8000")
    except Exception as e:
        print(f"[ERROR] Test Error: {str(e)}")

if __name__ == "__main__":
    test_dog_classifier_api()
