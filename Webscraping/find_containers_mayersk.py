import requests
import json
import pandas as pd

PATH = "C:/MIT/2025 SCALE Challenge"

# Define the URL
#url = "https://www.apmterminals.com/apm/api/trackandtrace/import-availability"

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
'''
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
'''

headers = {
    "accept": "application/json",
    "accept-language": "en,es;q=0.9,en-US;q=0.8,ru;q=0.7",
    "akamai-bm-telemetry":'a=63225E93142669566942EB4247DF079E&&&e=MDdFNTM3MTAwNkUxODIwNjA3NzkwNjUzQzQ1NzkxM0R+WUFBUXRyN0NGK3ByOHJXVEFRQUFzekl3dHhvakR4a1VTaE5CZzhWaXhBK1NmTFdTeUNQUHhvZU9BYmc5YmxaeXhqczgrVTBJVnJsRG9UZjRDQWxYelViWksvc3luemt5TU1HakdYcElOOVdZUFdhRFdKcnRwS1BTdTlzb0V4aGd3eldQY0tKZGJlSzR4NFJUV3FDRFJuQlg5cHBTbmJHR1hhRDJRWDRWK1djNkJCNUdROXpUYXlMcWtKREljU1hwUHdUeDRBSm5EK2M4UHlvMW5vaUM0dXhBc1gxRzAwR3JXcTF0L0V1ZkZFQWtJcTFXK3RuYjUzby9QejBWSlhEaGE2dkREbmd3SERaZVVqYU5FRlNvYTNFYlRxcXNKVVNaKzJvNHYvaXFlcTBZTUd6d1NaSTlvK0pXQ01IZmgyOFVQd3NvOXRSWmRGd3huemJDUlhQUHlINmdJYWQ4aUNXTVo3RUZpOUNsckhoUmk4Y0Z2VGl6dnRJblZ0Vk9GdGp1T2tKRFBwYm9yTWRYQzRERjZFZGNJTEpMfjM0MjA0ODR+MzI5NDI2MA==&&&sensor_data=MzsxOzI7MDszNDIwNDg0O3Rna2VpMjlMNTEzUWZoMFFITDRkcjdjS0g3R05UTW1DUlAwdlVwbVA3ZWs9OzEzLDEsMCwwLDQsNTQ7RTI6MSJicjQiWTdaTWUiS2RTIlp4dV8jcU5YIj1WcyI8ImgyOCwiYiJIPXkiUSY2ZU8icU9ZTyI/IisibUVPIlNaeCIrL1tyXyhiVDQiLWYleFgia2tqeD99SXR+b20idylpIn14Q1g6XThkXiwuRXZ4IWc2IjlZKFMiLkklMVhafDp2JkFzNUdZXkJoKG8xaiUwfWAxXTVDLkowUTlYIDAieyJkblYiQHt3W2dPQXZEWWwiJDdOKiJYLm4pZGQ0On4pXy5lRyF5WG08VjEiXTNCIiYiIixMLjlxUmZZPTsiIlAiPltMIk45OWtyOihkNlB5QCNBRnlAbyJoWiIqICUiOyFbIjYiZ2FtZyJiak4iUSBvImV8SS9wIkx7In4ifkRMRiBHOXEhdjY9TkYiMSE+Im5WNiJ1IiIlIkNJPyJpLyE6XyJqWyI6ImBnbzAiWDlgImY2TCJPST5EMzdiaSMiUGciNCtiKHwiM14tIkE2X0ZDInY5PX4iTSJFIiUtUSJ5NGsidm01P0JmfiIsMkAiJClAfFBOPVFHO05FIiZfRkoiclFNVnhoQ0wifVJqIlY0TWhMLX1hKXgiJjc2IjMiLkUwanojUls1Ry0yWiJsIjBIbSJZU0VjWCJ3Ln0iOHQvVjsiLTNhfCI6Ikw5R1REMVE+KWshcTNrRlVpeC9ua1YrQ0RocUhxMzN8TUJTdVA6I0wudElyR09weT5DSEhMOGpiX28gZHVaSCtqNkhOMSBLdHMjXSFKZEJsNWM1PixfJFJVczlpO3pWNmI8I1NeYUBrLixNU3xIdCJUcX4ibDUjIkNyLnZBN2pbK2srW19gRV9xXyI/WjQiSiIiQFJFIkpaWCIhXTliVjBBIkh3dytrQ1RRLDJ1SHtGITZmc3FQQU9TM1pQJXRjUz0yWTc3UEl1KS5neFgodEUoSG8kNn1hbCRsIikiWCA1IjsiPExrWEZSZkVgPXljO3lAJSI6IlFqeyJvUUdIPCl3ZSIgICJjTXJsTnAifmpYPyJvIiVmWHE6SC43bnZzP0BCPk1ZMmY2YjdJXU97KWdVKiggSjx4LiYud10samtTIjwiRm5JImdyKXlEInt3WyJQJDRiWyJ2MisiTyIiUSJHVTkielRPMWsoIkk0dy1fY00ieyIiRSJ8LyQiSVdfbCo4JiY8dXYiSlUiYyMvImxwZCJTInwqXi1BKERAWG0sND48PHRdUVpAbWBFUFNRRkk3MFg1TmM3a2xmJiliaio5XlZFK3coNnREQSU/LEY+SkVCcUxZfFtCS0JmYkZafTRQVFNAITdML3FzYHptNzI9NDp6VXgjdF1sczs6eDJte05STkQ1TGlKNj4zTE1uZzcoLXZOdDgubH4kXW9Wb0FkIEVMPC5EaFtAWkxoezdKcmF1Vk9uLi1yN0prOWE0cDZzcCx2c0F4WlFgXiE8Zm4pPlNEP159cmwqRiBELlA7XTdQbltuMWxSP1BsL1N6MlhWe21pOmFsbzhJe0kuX0t9V3M6LzR4O0cuSmQvUyk6ZCM+RDNrSkNNejpzPzR3YThvLlNiYU95d34yVTJQfkNyOFVhUzBnenJeeFtBKGlWLSY1dn44bD9OS3kxdDx3QHIxVWxvTCVGUiFYQHlmWUowekN7Jj4lcXB+QGxLIGYxcy5jW25fQFlpR3xuSFYtNCFuTngqRy9tLj92QHtEOGtNeEJifVpXcSldNClwI29zZk0tcCpRKyBIXzxRO3ZqQjRiM1FtSkJ1M3NMRzlOQUItLmtTajYqJUFTM3JhOzFtZzggLFM3P1QgdFdYVVBUb15dPi1bdCFyOUw8bmxISC97WUFkISR8SnxmXVJTZmp3dSpiWitWTlgjRihwZVRaQz1rY3hVW184YE83VUlsYigtQDN5W3QvLGMsdFtiTFBDPjB3RXd9LGI5LCM1OF5uKDQ1NThoRU9SM2leQ0E1SS5JLiBWLDJKLWFnUHgjSWMjITo6PyBgdCVPKyx3fnwkMEEkM1k5Q1tBJjAjSFV1PWR7QDI9JGwjMmxISDpQUXNhfW53WC9UYEo3R01wKVFmM1kgfC94WyY9fWVYYH0mSkBgRl80KkNpXUBEXSpFeTVlNWBwdl5PeTUmYW5wMjFmdCosKnNvMFhOO0ReLkg1VyxUdzpYQzFgIXhpenw9TXUmSVJrVUhkPz0sQlA4WDlgRnRSZiRqdjRwZFNxdjJXLz5jaiRvaTxxa242RXxKOW5VJWB9NTdDdEg4NlBkOlcmQXRvOjo6cEpCSG0ubEIvXk0zZTFKVFs3fHhzL0d+PHk7aCNIP1ApW2RyNF1YMHtXPCVlM15lelQvPUZjdVZ+VSRifVJJTzR5KDFfPXlfVzwpaFYqZ2x6ZEZYWzJHIFg/ck9lPkJQM3w7RXxaOyl5bHJPPnhUZXdxS2ZpPGVSJWgwLkl5IUs5IEhSO2NiPUpDQS0lSzlPZ1ZPc3hgd3FAKF5LKV1zMnJ1NksraGpWalZbTFIsbDJSSkdSIF0saVBBLCVIKzhgTkp7QCNqZ2RzaHFmaFlKai58e0dhRX1tU00/N2Y/Zi8sOVkjalVfXnhsLn0ge2AyTV1gLUs0dlpXRT1EdmsrT1dfNkpWOktQY2Z7fSZ4I0oqfCZhfG5STzhTQzd9ajhdciBTJn5vNCxKUXR3KDIqWjM/SHNMTiM5eTsjNHRjNngoOXhAQTNaYiY1XmIkIStXOE1XQXBzUFpZYm55ZVkkeyQ4eF5aaS0xT186SmhtdVJCTG9GenNwJi1EQU08SG9Yezd5Wmd6N0B1MmN+VlBfMSU5YUh9LygyS19tenl+WU9rIGpbb3xMXi5ZKkt+dTF2ZzRbP34oK0NYJik9O0k7IEJhZU5pdkdoQms/YkFIaE0/c0ExayExUmAgOFddX1VZe0tIKFVkNVtJcFMjV2l0enFAbl9kcW1KWSEvZ20reXlKZ3JiLkwhRDJgX3NWbjo1NnRESCtQTStOey1majEgIVssMT1uI2QsI1BDIV8hPzpLemBhWn0tZChgdVB2LCwsZjtPWyVSNl5fQiphTnFATVI4bXN2PWA0VT5eR1Z6dSt4TmJsO1tSKiZyZUo+UDdITDMhMChhJWIkbi51Q2d1fWNaYXBBIHFRVjk5b2QxeSxINXUhQHAybkQvWU90PU5hTz9Yck8seGJyVllIQW1XdSx5ZHUxeDUoWUZ+c0BtLEYqfD5XQiMhcXZhcFtaPHxLWlFEZyFjKi1jVTh+UzFIYWReIU8xb3ZkbXN0d3RRRXQrIH5FaUosemBPOzBmUGk7LCFjKGZLX09icCMgfVhZfVRHWnNBMl9RRkgwNW1UfUZESSNJQiNJQ09Pc2cja2w/YGVwT3BLOjMxNXx5X0QjUFdhMl9oVGRlOTleW3FaXjV1KiZoMHhwaV9fZmxYUXJIWWBDLHx2PS1fdStAWVVYOnd3KV5DPSU4My8+QSYtajxUbUYhMSZLVyU+a2YyMjlsbXQ0c0c9ODxDYmRoZFpCfDpJTCk7NVZgOEogQmhifVRFXSRXTEtDY19ydSguQHdYdDokdyp9ZXZTaT9aLCo+Mi5CYkA1MzpWbCEzTkxQOyZKfWtUb25MZkllS3FGWG5KT2Y6I3QqKj9lKjJSaGtnTXpAPyFSWi5VRmZTekRTKnB7R2tVTl5rNEYhMktgfm1iNGBrWSY3dzIuUjZyTlQgKiNTMzJ6P1BmMlR6P1JxbGs+eyRuSmlFeWEvIE5BUSQlL2U+RD5ackNuPlcoOGpaYz99My5fJGg6LGxmOn1Cbm4mXkJMTngwWzNqNWIjUFpZPHp+L2Q5KGBWOCp1VytNYnNuPVFYdUN4UytgR2IkPDslaSg+bzQ3cl9QSz8mYjBFV0w0QFJ8RSZURXRfJmZmKG9FcCNbOjJ4eGdrUEx6YiU/Jns0QntPMmBafHM0dzBOPzZIYEglJWt7c3ZoaEE0TGRjVGghYDQqZFI8eksvU2BaWyg6JmdzZHR0b2xoVUFdIyh0QltFb3NHQX0hU0xiJSN0RGBWRkJDXk1maHJLSHA6MjlrMCJLImBmTSJeOF4tcjBFImZGXSI8IiNXZkdOd0xKVk9SZEduPmsmVDsvJS0oODtsSjFGSjpFMy9TOWJqelBdMFhQQ3U7RCk1fnBwelQ2bE5LSlRfITwiZCJJWC4iTyIiWCIwMjQiWiJpKFVCTiJVRjgiPGB5Iiw3TlNFKmUiK1EzImFdYFFUYSwgYzxYbk48PT1qI1FvJkhXIls5OiIyIik8LkdGKiIiMyJdR2oiQSJ+NCJsR1QiQlpeMSIySGtJQTI8QyJmYCkiXjsoU1plQyJaMGkiaklOMisiLHZJaiJ2QmwiMEVtKCJjIiJkIlhWbyIkTS1VOl18QkoiMFciViIiUiJ2Ol4iUCJQQSIvQjEiWypWTCIkUHlrSWUiOV5aZUdgOmw7VC1+Lyx+KDMzPFRKXS8gPWAoZjBZPkF0RXplMUZ0OXxyPFtrOHBCMDU0OE0iaCJ0aDQieCIiUCI7VUoiYyk+KUEiLk1RZiJIIkoyTUMiaSIvVmUidyJoVzdBdyJ+byw1KSJiQDAiemZ7fHwiRiJYQjQhemB9Ojw1I0lEVjJ5IlMiQ2xLIjIiImsiS3giLCIiWyxMIkwwIjgiM0Z+QihqVy0iL2tzIntRICJWfH5LdW1tUSJ8S34iXiIibyJvUCkiNCIifCI9KnwiNyNUW1BmUW9GTisiNlZESSJVIiJSInFKOCIlKCJAMmciRDAvKjc2Njs9ImQ3WSIpIjwqU2FAbjIoMlVyNzc6amk5QDNOanYrWi9XaSNrYiF+eT1FfjNrfVJqKiYvXSIlYFIiU09mImYjWyIvaD4iISIiZk9SIjU2SCI8ZyI6SXYiQC1gIlVjdykyOSQiJjdWIg==',
    "api-version": "v2",
    "consumer-key": "UtMm6JCDcGTnMGErNGvS2B98kt1Wl25H",
    "dnt": "1",
    "origin": "https://www.maersk.com",
    "priority": "u=1, i",
    "referer": "https://www.maersk.com/",
    "sec-ch-ua": '"Google Chrome";v="129", "Not=A?Brand";v="8", "Chromium";v="129"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "sec-gpc": "1",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
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
    #"MSKU9869433"# 
    '''
    data = {
        "TerminalId": "cfc387ee-e47e-400a-80c5-85d4316f1af9",
        "DateFormat": "mm/dd/y",
        "DatasourceId": "0214600e-9b26-46c2-badd-bd4f3a295e13",
        #"Ids": ["TRHU7691752", "TRHU7691751","TRHU7691753"]
        "Ids": [f"{cnt}"]
    }
    '''
    #print(data)

    
    try:
    # Make the POST request with a timeout
        url = f"https://api.maersk.com/synergy/tracking/{cnt}?operator=MAEU"
        response = requests.get(url, headers=headers)
        print(f"url is {url}")
        # Check the response status and content
        if response.status_code == 200:
            #print("Request successful")
            json_data = response.json()
            print(json_data)
            exit()
            '''
            if json_data.get("ResultsCount", 0) > 0:
                print(f"putting k = {k} ")
                api_responses[container] = json_data
            '''
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


