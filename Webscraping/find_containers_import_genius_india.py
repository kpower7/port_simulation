import requests
import json
import pandas as pd

PATH = "C:\\Users\\k_pow\\OneDrive\\Documents"

# Define the URL
# Find using Network in developer on browser
# Check off "Preserve Log" and "Invert"
# Find the "Import" key and use that url

url = "https://7dsjt4q1af.execute-api.us-east-1.amazonaws.com/clients/94169/searches/inf/import"
# Define the headers including 'site-id', 'user-agent', and 'x-requested-with'
headers = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "en,es;q=0.9,en-US;q=0.8,ru;q=0.7",
    #update authorization bearer
    "authorization": "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Im5NTDluaTEtZ01CTW54Y3Jybl9PcyJ9.eyJodHRwczovL2ltcG9ydGdlbml1cy5jb20vcHJvZmlsZSI6eyJjbGllbnRJZCI6OTQxNjksImhhc1Nzb09ubHkiOnRydWUsInJlY3VybHlBY2NvdW50Q29kZSI6ImtldnBvd2VyQG1pdC5lZHUifSwibmlja25hbWUiOiJrZXZwb3dlciIsIm5hbWUiOiJrZXZwb3dlckBtaXQuZWR1IiwicGljdHVyZSI6Imh0dHBzOi8vcy5ncmF2YXRhci5jb20vYXZhdGFyLzEzMzIzZDlhYTc4MDE0ZDVkMjc1MjI3ZDI0MmNhZjQzP3M9NDgwJnI9cGcmZD1odHRwcyUzQSUyRiUyRmNkbi5hdXRoMC5jb20lMkZhdmF0YXJzJTJGa2UucG5nIiwidXBkYXRlZF9hdCI6IjIwMjQtMTItMTFUMTc6MTk6NDQuODQxWiIsImVtYWlsIjoia2V2cG93ZXJAbWl0LmVkdSIsImlzcyI6Imh0dHBzOi8vbG9naW4uaW1wb3J0Z2VuaXVzLmNvbS8iLCJhdWQiOiIxYWRTMHg5Mm5YMkl1bVhmbm5kcHlvbDREakY0WGhQbCIsImlhdCI6MTczMzkzNzU5MiwiZXhwIjoxNzM0ODAxNTkyLCJzdWIiOiJhdXRoMHw5NDE2OSIsInNpZCI6IlExbDdDQmNXb25oQk5HdG1WN3hlQVZiYS1aNlI1OWRsIiwibm9uY2UiOiJUY2gyT0NycTBjIn0.IzqlnjIFeYeds704ErwqH2ciSnvdAE_oqffeqzfP1Fqnl-8s4obhsaiS-GqegAFmmPAjj97p6J-tO8_yJESp3SKacCy9uiPrsNWyuX6uDVXcAYJp7AeNcKPV2gq0W8Put15AvlpAcqEowzDnr08sDbGU3ZPSQV2Go3ImH05f_Mnskbe7Oycolx87PRlEW2lUV-nbz34HBL0DbsmhN86CdIkzPj_JCJmMecEHUHxRfH7gRP_sQkZUAZL0We50ane0FO1ifZ9nC1Ks5aSoYnyRd_c9y3-xE5bwUrEGPHOFil6j0eC5P3xvq-FusRl7up2rKRPNAGs3Rt_mb3Z8wXHBog",
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
page_number = 5
for container in range(page_number):
    k += 1
    
    #change search parameters for India
    data = {"searchId":23441908,"pagination":{"page":k,"pageSize":100},"sortBy":{"field":"arrivalDate","direction":"DESC"},"contextId":"ccd80d62f75941d701f73707c0107b84","params":{"filter":[{"operator":"and","type":"text","input":{"keyword":"New York","modifier":"contains","field":"usPort"}}],"searchDates":[1733011200, 1733193599],"searchOptions":{}},"showGlobalData":[]}

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
df.to_csv(f"{PATH}/Results_India.csv")


