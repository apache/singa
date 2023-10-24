
import requests

url = 'http://localhost:8093/'
columns = ['label', 'col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10']
response = requests.post(
    url, json={'columns': columns,
               'name_space': "train",
               'table_name': "frappe_train",
               "batch_size": 32})
print(response.json())

response = requests.post(
    url, json={'columns': columns,
               'name_space': "valid",
               'table_name': "frappe_valid",
               "batch_size": 1024})
print(response.json())

