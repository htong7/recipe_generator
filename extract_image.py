import os
import requests
import base64
from openai import OpenAI


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image(image_path):
    client = OpenAI()
    base64_image = encode_image(image_path)
    prompt = '''
    Identify all the food, beverage, and condiment items visible in the image.
    Provide a Python list of all the items visible in the image. 
    Try to identify those items if there are labels on them.
    Please only include those items you're absolutely sure of, without 
    any additional descriptions of the ingredients. Just the ingredient names, please.
    If the classification probability of the item is less than 90%, exclude them from the list.
    '''
    

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                },
                },
            ],
            }
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content

def main():
    image_path = '/Users/felixtong/Desktop/borealis/image_2_recipe/photo3.jpg'
    result = analyze_image(image_path)
    if result:
        print("Image Analysis Result:")
        print(result)

if __name__ == "__main__":
    main()
