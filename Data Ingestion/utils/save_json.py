# utils/save_json.py
import json

# Function to save data to a JSON file
def save_to_json(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
    
    print(f"Data successfully saved to {output_file}")
