import requests


response = requests.get("http://localhost:5000/result/Untitled.FR12_Page_017.jpg", headers={'Content-Type': 'application/json'}) # , 'Authorization':token

print(response.text)