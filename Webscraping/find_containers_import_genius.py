import requests
import json
import pandas as pd

PATH = "C:/MIT/2025 SCALE Challenge"

url = "https://7dsjt4q1af.execute-api.us-east-1.amazonaws.com/clients/88135/searches/us/import"
# Define the headers including 'site-id', 'user-agent', and 'x-requested-with'
headers = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "en,es;q=0.9,en-US;q=0.8,ru;q=0.7",
    "authorization": "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Im5NTDluaTEtZ01CTW54Y3Jybl9PcyJ9.eyJodHRwczovL2ltcG9ydGdlbml1cy5jb20vcHJvZmlsZSI6eyJjbGllbnRJZCI6ODgxMzUsImhhc1Nzb09ubHkiOnRydWUsInJlY3VybHlBY2NvdW50Q29kZSI6Im5hcmlzdG92QG1pdC5lZHUifSwibmlja25hbWUiOiJuYXJpc3RvdiIsIm5hbWUiOiJuYXJpc3RvdkBtaXQuZWR1IiwicGljdHVyZSI6Imh0dHBzOi8vcy5ncmF2YXRhci5jb20vYXZhdGFyLzJmZjhiZTUwOGE0MzJjZjIyMTA1ZjI1OGEwZjE0YjNjP3M9NDgwJnI9cGcmZD1odHRwcyUzQSUyRiUyRmNkbi5hdXRoMC5jb20lMkZhdmF0YXJzJTJGbmEucG5nIiwidXBkYXRlZF9hdCI6IjIwMjQtMTItMTFUMTc6MjU6MTEuNjMzWiIsImVtYWlsIjoibmFyaXN0b3ZAbWl0LmVkdSIsImlzcyI6Imh0dHBzOi8vbG9naW4uaW1wb3J0Z2VuaXVzLmNvbS8iLCJhdWQiOiIxYWRTMHg5Mm5YMkl1bVhmbm5kcHlvbDREakY0WGhQbCIsImlhdCI6MTczMzkzNzkxNCwiZXhwIjoxNzM0ODAxOTE0LCJzdWIiOiJhdXRoMHw4ODEzNSIsInNpZCI6Ik5IMzVxTUpNV0tPaTdVc2pJc3RwZzBrbV96ei1Tc2JDIiwibm9uY2UiOiJVMGR2TTFUSkRMIn0.INHmrxbC44cqS9z3Vt4ahholGiMJIPUfYJr6xlrxE3lmDJ-SonGzEarb2o1Nfgs2ZsmOOZ-xpracx3Uom5i4dUT83xwccFhfx0l0bkfaoqf48bvTBY5GyZKUqmzGUVsrScqoFkmsxmQsBgjr0Uu1BKtr0qt-8IpQznCRebYK2i4Lmq4sSJND-E5ukIOCOgypRT5E4U9ukjQifVFn7Ct6pqB3DC9xQDQwP_2LfejxvVVGWa0cdBpwtQvEiqzoi1vBCG-UILeBrlPXgC8DUBsZntweZHWEceI2iHfOr-ppBGSfomUBJvwMRvYmGM2QSMTdqVP-iNdr6T1krGmPIwYhpg",
    "content-type": "application/json",
    "dnt": "1",
    "origin": "https://app.importgenius.com",
    "priority": "u=1, i",
    "referer": "https://app.importgenius.com/",
    "sec-ch-ua": "\"Google Chrome\";v=\"129\", \"Not=A?Brand\";v=\"8\", \"Chromium\";v=\"129\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"Windows\"",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "cross-site",
    "sec-gpc": "1",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
}

# Define the raw data to send in the request

#df = pd.read_csv(f"{PATH}/NY20241208.csv",  index_col=False)

#df = df[["CONTAINER NUMBER", "COUNTRY OF ORIGIN", "CARRIER CODE", "CARRIER NAME"]]

# Explode the 'CONTAINER NUMBER' column
#df['CONTAINER NUMBER'] = df['CONTAINER NUMBER'].str.split()  # Split by space into lists
#df_exploded = df.explode('CONTAINER NUMBER').reset_index(drop=True)

#print(df_exploded.info)

#HASU5054644

#unique_containers = df_exploded[df_exploded['CARRIER CODE'] == 'MAEU']['CONTAINER NUMBER'].unique()
timeout = 10

# Dictionary to store API responses
api_responses = {}
k = 0
all_data = []
page_number = 620 #(6197 // 100) + 1
for container in range(page_number):
    k += 1
    
    data = {"searchId":23423942,"pagination":{"page":k,"pageSize":100},"sortBy":{"field":"arrivalDate","direction":"DESC"},"contextId":"ccd80d62f75941d701f73707c0107b84","params":{"filter":[{"operator":"and","type":"text","input":{"keyword":"New York","modifier":"contains","field":"usPort"}}],"searchDates":[1733270400,1734048000],"searchOptions":{}},"showGlobalData":[]}

    formatted_data = json.dumps(data, separators=(',', ':'))
    #print(formatted_data)
    #exit()
    
    try:
    # Make the POST request with a timeout
        #request = requests.Request("POST", url, headers=headers, json=formatted_data)

        # Prepare the request
        #prepared = request.prepare()

        # Print the prepared request details
        #print("URL:", prepared.url)
        #print("Headers:", prepared.headers)
        #print("Body:", prepared.body)
        #exit()
        response = requests.post(url, headers=headers, json=data)#, timeout=timeout)
        
        # Check the response status and content
        if response.status_code == 200:
            #print("Request successful")
            json_data = response.json()
            #if json_data.get("ResultsCount", 0) > 0:
            print(f"putting k = {k}")
            json_response = response.json()
            data = json_response.get("data", [])
            all_data.extend(data)  #             
                        
        else:
            print(f"Request failed with status code {response.status_code}")
            print(response.text)

    except requests.exceptions.Timeout:
        print(f"Request timed out after {timeout} seconds")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
#print(api_responses)
#result_df = process_responses_to_df(api_responses)
df = pd.DataFrame(all_data)
df.to_csv(f"{PATH}/Results_ig4.csv")


