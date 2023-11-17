import requests
import os

# response = requests.get("http://localhost:5000/result/Untitled.FR12_Page_017.jpg", headers={'Content-Type': 'application/json'}) # , 'Authorization':token

# print(response.text)

severURL = 'http://127.0.0.1:5000'
url = severURL + '/ElecBills'
# payload = {
#     'fileName' : batchName
# }
inputs = os.listdir("inputs")
allfiles = [v for v in inputs if v[-4:] in ['.jpg', '.png']]
outputs = {}
for filePath in allfiles:
    files = [('files', open(filePath, 'rb'))]
    response = requests.post(url, files=files)

    print(response.json())