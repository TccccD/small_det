import json
import numpy as np


class Model(object):

    def predict(self, x):
        class_id = np.random.randint(0, 50)
        score = np.random.randint(0, 50)

        return [[class_id, score, [0, 0, 0, 0]]]


block = {"image_id": 0,
         "category_id": 0,
         "bbox": [0, 0, 0, 0],
         "score": 0.1}

model = Model()
data_val = "./instances_val2017.json"
data = json.load(open(data_val, "r", encoding="utf-8"))
image_id = []
classes = [x["name"] for x in data["categories"]]
classes.insert(0, "None")
res_submit = []
for image in data["images"]:
    image_id = image["id"]
    image_path = image["file_name"]
    res = model.predict(image_path)
    for class_id, score, bbox in res:
        block["image_id"] = image_id
        block["category_id"] = class_id
        block["score"] = score
        block["bbox"] = bbox
        res_submit.append(block)

with open("./test_submit.json", "w") as f:
    f.write(json.dumps(res_submit))


