import requests

url = 'http://localhost:5000/results'
r = requests.post(url, json={'sentences':['good job.']})

print(r.json())