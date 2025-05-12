import json
import os
import re
import numpy as np
import json
import random
from datasets import Dataset
import  argparse
import torch
import json
import re

def extract_data(root_path):

    with open(root_path, 'r') as f :#, \    
        train_file = json.load(f)

    test_data=dict()

    # TODO: convert the data into test data, containing the importance, multiple choice questions, graph questions
    for scene_id in train_file.keys():
        scene_data = train_file[scene_id]['key_frames']
        
        # for test file
        test_data[scene_id] = dict()
        test_data[scene_id]['key_frames'] = dict()

        for frame_id in scene_data.keys():
            frame_data_infos = scene_data[frame_id]['key_object_infos']
            frame_data_qa = scene_data[frame_id]['QA']
            image_paths = scene_data[frame_id]['image_paths']

            # for test file
            test_data[scene_id]['key_frames'][frame_id] = dict()
            # test_data[scene_id]['key_frames'][frame_id]['key_object_infos'] = frame_data_infos
            test_data[scene_id]['key_frames'][frame_id]['QA'] = dict()
            test_data[scene_id]['key_frames'][frame_id]['image_paths'] = image_paths
            test_data[scene_id]['key_frames'][frame_id]['QA']['perception'] = []
            test_data[scene_id]['key_frames'][frame_id]['QA']['prediction'] = []
            test_data[scene_id]['key_frames'][frame_id]['QA']['planning'] = []
            test_data[scene_id]['key_frames'][frame_id]['QA']['behavior'] = []

            # get the classes of the important objects
            classes = []
            for obj_id in frame_data_infos.keys():
                obj_data = frame_data_infos[obj_id]
                classes.append(obj_data['Visual_description'].split('.')[0])
            print(classes)

            # get the location of the important objects
            locations = []
            for obj_id in frame_data_infos.keys():
                locations.append(obj_id)
            print(locations)

            # get the questions and answers of the perception
            perception = frame_data_qa["perception"]
            prediction = frame_data_qa["prediction"]
            planning = frame_data_qa["planning"]
            behavior = frame_data_qa["behavior"]

            for qa in perception:
                question = qa['Q']
                answer = qa['A']

                # according to the classes to select the corresponding question
                flag = 1
                for cl in classes:
                    if cl.lower() not in answer.lower():
                        flag = 0
                if flag == 1:
                    qa['tag'] = [2]
                    test_data[scene_id]['key_frames'][frame_id]['QA']['perception'].append(qa)
                    break
                
            # get the multiple choice questions and answers
            for qa in perception:
                question = qa['Q']
                answer = qa['A']
                if "What is the moving status of object".lower() in question.lower():
                    qa['tag'] = [0]
                    test_data[scene_id]['key_frames'][frame_id]['QA']['perception'].append(qa)
                    break
            
            # get the graph questions and answers
            for qa in prediction:
                question = qa['Q']
                answer = qa['A']

                # according to the location to select the corresponding question
                flag = 1
                for loc in locations:
                    if loc.lower() not in answer.lower():
                        flag = 0
                if flag == 1:
                    qa['tag'] = [3]
                    test_data[scene_id]['key_frames'][frame_id]['QA']['prediction'].append(qa)
                    break

            # get the yes or no questions and answers
            for qa in prediction:
                question = qa['Q']
                answer = qa['A']
                if "yes" in answer.lower() or "no" in answer.lower():
                    qa['tag'] = [0]
                    test_data[scene_id]['key_frames'][frame_id]['QA']['prediction'].append(qa)
                    break

            # get the three questions from the planning "safe actions", "collision", ""
            actions_question_added = False
            collision_question_added = False
            safe_actions_question_added = False
            for qa in planning:
                question = qa['Q']
                answer = qa['A']
                if "What actions could the ego vehicle take".lower() in question.lower() and not actions_question_added:
                    qa['tag'] = [1]
                    test_data[scene_id]['key_frames'][frame_id]['QA']['planning'].append(qa)
                    actions_question_added = True
                if "lead to a collision" in question.lower() and not collision_question_added:
                    qa['tag'] = [1]
                    test_data[scene_id]['key_frames'][frame_id]['QA']['planning'].append(qa)
                    collision_question_added = True
                if "safe actions" in question.lower() and not safe_actions_question_added:
                    qa['tag'] = [1]
                    test_data[scene_id]['key_frames'][frame_id]['QA']['planning'].append(qa)
                    safe_actions_question_added = True

                # Check if all question types have been added and exit the loop
                if actions_question_added and collision_question_added and safe_actions_question_added:
                    break
            
            for qa in behavior:
                question = qa['Q']
                answer = qa['A']
                qa['tag'] = [0]
                test_data[scene_id]['key_frames'][frame_id]['QA']['behavior'].append(qa)

    return test_data


def rule_based1(question, answer):
    rule = ["Going ahead.", "Turn right.", "Turn left.", "Stopped.", "Back up.", "Reverse parking.", "Drive backward."]
    rule.remove(answer)
    choices = random.sample(rule, 3)
    choices.append(answer)
    random.shuffle(choices)
    idx = choices.index(answer)
    question += f" Please select the correct answer from the following options: A. {choices[0]} B. {choices[1]} C. {choices[2]} D. {choices[3]}"
    mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
    return {"Q": question, "A": mapping[idx]}

def rule_based2(question, answer):
    rule = ['The ego vehicle is slightly steering to the left. The ego vehicle is driving very fast.', 'The ego vehicle is steering to the left. The ego vehicle is driving with normal speed.', 'The ego vehicle is steering to the left. The ego vehicle is driving fast.', 'The ego vehicle is slightly steering to the right. The ego vehicle is driving fast.', 'The ego vehicle is going straight. The ego vehicle is driving slowly.', 'The ego vehicle is going straight. The ego vehicle is driving with normal speed.', 'The ego vehicle is slightly steering to the left. The ego vehicle is driving with normal speed.', 'The ego vehicle is slightly steering to the left. The ego vehicle is driving slowly.', 'The ego vehicle is slightly steering to the right. The ego vehicle is driving slowly.', 'The ego vehicle is slightly steering to the right. The ego vehicle is driving very fast.', 'The ego vehicle is steering to the right. The ego vehicle is driving fast.', 'The ego vehicle is steering to the right. The ego vehicle is driving very fast.', 'The ego vehicle is slightly steering to the left. The ego vehicle is driving fast.', 'The ego vehicle is steering to the left. The ego vehicle is driving very fast.', 'The ego vehicle is going straight. The ego vehicle is not moving.', 'The ego vehicle is slightly steering to the right. The ego vehicle is driving with normal speed.', 'The ego vehicle is steering to the right. The ego vehicle is driving slowly.', 'The ego vehicle is steering to the right. The ego vehicle is driving with normal speed.', 'The ego vehicle is going straight. The ego vehicle is driving very fast.', 'The ego vehicle is going straight. The ego vehicle is driving fast.', 'The ego vehicle is steering to the left. The ego vehicle is driving slowly.']
    rule.remove(answer)
    choices = random.sample(rule, 3)
    choices.append(answer)
    random.shuffle(choices)
    idx = choices.index(answer)
    question += f" Please select the correct answer from the following options: A. {choices[0]} B. {choices[1]} C. {choices[2]} D. {choices[3]}"
    mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
    return {"Q": question, "A": mapping[idx]}
    

def loop_test(test_file):


    for scene_id in test_file.keys():
        scene_data = test_file[scene_id]['key_frames']

        for frame_id in scene_data.keys():
            # frame_data_infos = scene_data[frame_id]['key_object_infos']
            frame_data_qa = scene_data[frame_id]['QA']
            image_paths = scene_data[frame_id]['image_paths']

            test_file[scene_id]['key_frames'][frame_id] = dict()
            # test_file[scene_id]['key_frames'][frame_id]['key_object_infos'] = frame_data_infos
            test_file[scene_id]['key_frames'][frame_id]['QA'] = dict()
            test_file[scene_id]['key_frames'][frame_id]['QA']['perception'] = []
            # add all prediction and planning
            test_file[scene_id]['key_frames'][frame_id]['QA']['prediction'] = frame_data_qa["prediction"]
            test_file[scene_id]['key_frames'][frame_id]['QA']['planning'] = frame_data_qa["planning"]

            test_file[scene_id]['key_frames'][frame_id]['QA']['behavior'] = []
            test_file[scene_id]['key_frames'][frame_id]['image_paths'] = image_paths

            for qa in frame_data_qa["perception"]:
                question = qa['Q']
                answer = qa['A']
                if "What is the moving status of object".lower() in question.lower():
                    qa.update(rule_based1(question, answer))
                    test_file[scene_id]['key_frames'][frame_id]['QA']['perception'].append(qa)
                else:
                    test_file[scene_id]['key_frames'][frame_id]['QA']['perception'].append(qa)

            for qa in frame_data_qa["behavior"]:
                question = qa['Q']
                answer = qa['A']
                qa.update(rule_based2(question, answer))
                test_file[scene_id]['key_frames'][frame_id]['QA']['behavior'].append(qa)

    return test_file

def split_by_key_frame(test_file, train_ratio=0.8, seed=42): 
    frame_datas = []
    for scene_id in test_file.keys():
        scene_data = test_file[scene_id]['key_frames']

        for frame_id in scene_data.keys():
            image_paths = scene_data[frame_id]['image_paths']
            image_paths = [image_paths[key].replace("..", "data/DriveLM_nuScenes") for key in image_paths.keys()]

            frame_data = scene_data[frame_id]
            frame_data['image_paths'] = image_paths
            frame_data.update({'scene_id': scene_id, 'frame_id': frame_id})
            frame_datas.append(frame_data)
    # 根据给定随机种子打乱所有关键帧数据
    random.seed(seed)
    random.shuffle(frame_datas)
    # 根据 train_ratio 计算划分索引
    split_idx = int(len(frame_datas) * train_ratio)
    train_frames = frame_datas[:split_idx]
    val_frames = frame_datas[split_idx:]
    return train_frames, val_frames


def convert2vlm(test_file):
    output = []
    for frame_data in test_file:

        image_paths = frame_data['image_paths']

        frame_data_qa = frame_data['QA']
        QA_pairs = frame_data_qa["perception"] + frame_data_qa["prediction"] + frame_data_qa["planning"] + frame_data_qa["behavior"]
        
        for idx, qa in enumerate(QA_pairs):
            question = qa['Q']
            answer = qa['A']
            output.append(
                {
                    "id": frame_data['scene_id'] + "_" + frame_data['frame_id'] + "_" + str(idx),
                    "image": image_paths,
                    "conversations": [
                        {
                            "from": "human",
                            "value": question
                        },
                        {
                            "from": "gpt",
                            "value": answer
                        },
                    ]
                }
            )

    return output


def convert_to_hf_dataset(json_data):
    """Convert JSON data to Hugging Face Dataset format."""
    hf_data = []
    for item in json_data:
        hf_data.append({
            "id": item["id"],
            "image_paths": item["image"],
            "conversations": item["conversations"],
        })
    return Dataset.from_list(hf_data)


# 坐标变换函数
def rescale_coords(x, y, orig_size=(1600, 900), target_size=(224, 224)):
    new_x = x / orig_size[0] * target_size[0]
    new_y = y / orig_size[1] * target_size[1]
    return round(new_x, 2), round(new_y, 2)

# 替换字符串中形如 <c1,CAM_FRONT_RIGHT,1116.7,432.5> 的坐标
def replace_coords_in_text(text):
    pattern = r"<([^,]+),([^,]+),([0-9.]+),([0-9.]+)>"
    
    def repl(match):
        name, cam, x, y = match.groups()
        x_new, y_new = rescale_coords(float(x), float(y))
        return f"<{name},{cam},{x_new},{y_new}>"
    
    return re.sub(pattern, repl, text)


# 主处理函数
def convert_coors_system(data):
    for item in data:
        qa = item.get("QA", {})
        for _, qa_list in qa.items():
            for qa_item in qa_list:
                if "Q" in qa_item and isinstance(qa_item["Q"], str):
                    qa_item["Q"] = replace_coords_in_text(qa_item["Q"])
                if "A" in qa_item and isinstance(qa_item["A"], str):
                    qa_item["A"] = replace_coords_in_text(qa_item["A"])
    return data

def create_drivelm_nus(args):
    # extract the data from the training json file
    test_data = extract_data(args.src)

    # convert data: transform the obtained test.json data into the required test format
    rule_data = loop_test(test_data)

    # convert data into llama-adapter format
    train_frames, val_frames = split_by_key_frame(rule_data)

    # convert pixel valua coordinates system
    train_frames = convert_coors_system(train_frames)
    val_frames = convert_coors_system(val_frames)

    # 将转换后的数据保存到文件中
    os.makedirs("data/DriveLM_nuScenes/refs/",exist_ok=True)
    
    with open("data/DriveLM_nuScenes/refs/train_cot.json", "w", encoding="utf-8") as f:
        json.dump(train_frames, f, ensure_ascii=False, indent=2)

    with open("data/DriveLM_nuScenes/refs/val_cot.json", "w", encoding="utf-8") as f:
        json.dump(val_frames, f, ensure_ascii=False, indent=2)

    # convert to HF Dataset style
    output_train = convert2vlm(train_frames)
    output_val = convert2vlm(val_frames)
    with open("data/DriveLM_nuScenes/refs/val_qa_style.json", "w", encoding="utf-8") as f:
        json.dump(output_val, f, ensure_ascii=False, indent=2)

    # convert to HF style
    train_dataset = convert_to_hf_dataset(output_train)
    val_dataset = convert_to_hf_dataset(output_val)

    # Save train and validation datasets separately
    train_dataset.save_to_disk(args.train_data) 
    val_dataset.save_to_disk(args.val_data)       
    print("finished...")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=str, default=None, help='the json file download from DriveLM-nuScenes repo website')
    parser.add_argument("--resize_tgt", type=str, default=224, help='the pixel coordinate system to be transformed to')
    parser.add_argument("--train_data", type=str, default='data/DriveLM_nuScenes/split/train/', help='the huggingface Dataset style data')
    parser.add_argument("--val_data", type=str, default='data/DriveLM_nuScenes/split/val/', help='the huggingface Dataset style data')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    create_drivelm_nus(args)


