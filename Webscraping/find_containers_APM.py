import requests
import json
import pandas as pd

PATH = "C:/MIT/2025 SCALE Challenge"

# Define the URL
url = "https://www.apmterminals.com/apm/api/trackandtrace/import-availability"

# Function to process API responses into a DataFrame
def process_responses_to_df(api_responses):
    rows = []
    print(api_responses)
    for container, response in api_responses.items():
        if response and "ContainerAvailabilityResults" in response:
            for result in response["ContainerAvailabilityResults"]:
                result["CONTAINER NUMBER"] = container
                rows.append(result)
    return pd.DataFrame(rows)

# Define the headers including 'site-id', 'user-agent', and 'x-requested-with'
headers = {
    "accept": "application/json, text/javascript, */*; q=0.01",
    "accept-language": "en,es;q=0.9,en-US;q=0.8,ru;q=0.7",
    "content-type": "application/json",
    "dnt": "1",
    "origin": "https://www.apmterminals.com",
    "priority": "u=1, i",
    "referer": "https://www.apmterminals.com/en/port-elizabeth/track-and-trace/import-availability",
    "sec-ch-ua": "\"Google Chrome\";v=\"129\", \"Not=A?Brand\";v=\"8\", \"Chromium\";v=\"129\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"Windows\"",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "sec-gpc": "1",
    "site-id": "cfc387ee-e47e-400a-80c5-85d4316f1af9",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
    "x-requested-with": "XMLHttpRequest"
}

# Define the raw data to send in the request

df = pd.read_csv(f"{PATH}/NY20241208.csv",  index_col=False)

df = df[["CONTAINER NUMBER", "COUNTRY OF ORIGIN", "CARRIER CODE", "CARRIER NAME"]]

# Explode the 'CONTAINER NUMBER' column
df['CONTAINER NUMBER'] = df['CONTAINER NUMBER'].str.split()  # Split by space into lists
df_exploded = df.explode('CONTAINER NUMBER').reset_index(drop=True)

print(df_exploded.info)

#HASU5054644

unique_containers = df_exploded[df_exploded['CARRIER CODE'] == 'MAEU']['CONTAINER NUMBER'].unique()
timeout = 10

# Dictionary to store API responses
api_responses = {}
k = 0
for container in unique_containers:
    k += 1
    cnt = container # "HASU5054644" #container
    data = {
        "TerminalId": "cfc387ee-e47e-400a-80c5-85d4316f1af9",
        "DateFormat": "mm/dd/y",
        "DatasourceId": "0214600e-9b26-46c2-badd-bd4f3a295e13",
        #"Ids": ["TRHU7691752", "TRHU7691751","TRHU7691753"]
        "Ids": [f"{cnt}"]
    }
    #print(data)

    
    try:
    # Make the POST request with a timeout
        response = requests.post(url, headers=headers, json=data, timeout=timeout)
        
        # Check the response status and content
        if response.status_code == 200:
            #print("Request successful")
            json_data = response.json()
            if json_data.get("ResultsCount", 0) > 0:
                print(f"putting k = {k}")
                api_responses[container] = json_data
            
        else:
            print(f"Request failed with status code {response.status_code}")
            print(response.text)

    except requests.exceptions.Timeout:
        print(f"Request timed out after {timeout} seconds")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
print(f"k is {k}")
result_df = process_responses_to_df(api_responses)
result_df.to_csv(f"{PATH}/Results.csv")


