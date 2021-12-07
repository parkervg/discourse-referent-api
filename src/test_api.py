import requests
import json


if __name__ == "__main__":
    text = "Today I am going to plant a tree. I take my shovel so I can dig a hole first. Then, "
    r = requests.get("http://127.0.0.1:8000/get_json_prediction/", json={"text": text})
    print(json.dumps(json.loads(r.json()), indent=4))
