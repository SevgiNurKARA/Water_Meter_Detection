import requests

# API URL
url = 'http://127.0.0.1:5000/predict'

# Local paths to the image and model
file_path = r'C:\Users\sevgi\OneDrive\Masaüstü\sayac\test\not_water_meter\bird.jpg'
model_path = 'my_model.keras'

# Prepare the files and data for the POST request
with open(file_path, 'rb') as f:
    files = {'file': f}
    data = {'model_path': model_path}

    # Send POST request to the API
    response = requests.post(url, files=files, data=data)

    # Print the response from the API
    if response.status_code == 200:
        print(response.json())
    else:
        print(f"Error: {response.status_code}")
        print(response.json())
