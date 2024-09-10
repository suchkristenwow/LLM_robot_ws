import requests 
import os 
import logging 
from requests.exceptions import SSLError, ConnectionError 
import time 
import base64

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def generate_with_openai(prompt, conversation_history=None, max_retries=25, retry_delay=10, n_predict=2048, temperature=0.9, top_p=0.9, image_path=None):
    print("Trying to get an answer from ChatGPT, hold on ...")
    attempts = 0
    openai_api_key = os.getenv("openai_key")
    
    if conversation_history is None:
        conversation_history = [
            {
                "role": "system",
                "content": "You are a helpful, respectful, and honest assistant."
            }
        ]
    
    conversation_history.append({
        "role": "user",
        "content": prompt
    })

    if image_path is None:
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + str(openai_api_key) 
        }

        data = {
            "model": "gpt-4",
            "messages": conversation_history,
            "max_tokens": n_predict,
            "temperature": temperature,
            "top_p": top_p,
        }
    else:
        base64_image = encode_image(image_path)
        conversation_history.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt 
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,"+str(base64_image) 
                    }
                }
            ]
        })

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + str(openai_api_key)
        }

        data = {
            "model": "gpt-4-turbo",
            "messages": conversation_history,
            "max_tokens": 300
        }

    response = None

    while attempts < max_retries:
        print("Trying to get response ...")
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions", json=data, headers=headers
            )
            if response.status_code == 200:
                response_content = response.json()["choices"][0]["message"]["content"]
                conversation_history.append({
                    "role": "assistant",
                    "content": response_content
                })
                return response_content, conversation_history
            else:
                #logging.error(f"Error generating text with OpenAI: {response.text}")
                logging.error("Error generating text with OpenAI: " + str(response.text))
                return None, conversation_history
        except (SSLError, ConnectionError) as e:
            # print(f"Encountered error: {e}")'
            print("Encountered error: " + str(e)) 
            attempts += 1
            if attempts < max_retries:
                #print(f"Retrying in {retry_delay} seconds...")
                print("Retrying in "+ string(retry_delay) + " seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Unable to complete request.")
                raise
        except Exception as e:
            #print(f"Unexpected error: {e}")
            print("Unexpected error: "+str(e))
            raise