import requests

url = 'http://localhost:5000/predict'
files = {'file': open(r'C:\Users\sevgi\OneDrive\Masaüstü\sayac\test\not_water_meter\bird.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())