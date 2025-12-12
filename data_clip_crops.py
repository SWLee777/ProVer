import os
import cv2
import torch
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms as T
from cn_clip.clip import load_from_name, available_models

def load_yolov8():
    """加载YOLOv8模型"""
    model = YOLO("yolov8n.pt")  # 使用nano版本
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model


def detect_and_crop(image, yolo_model, min_size=32, min_score=0.3, max_crops=5):
    """
    使用YOLOv8检测并裁剪图像
    """
    # 统一转换为PIL格式处理
    if isinstance(image, np.ndarray):  # OpenCV格式(BGR)
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:  # 假定是PIL格式
        image_pil = image.convert('RGB')

    # 执行检测
    results = yolo_model(image_pil)

    # 获取检测结果(NumPy数组)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # [N,4]格式检测框
    scores = results[0].boxes.conf.cpu().numpy()  # 置信度

    # 过滤检测结果
    keep = (scores >= min_score) & (box_area(boxes) >= min_size ** 2)
    boxes = boxes[keep]
    scores = scores[keep]

    # 按置信度排序
    sorted_indices = np.argsort(scores)[::-1]
    boxes = boxes[sorted_indices]

    # 准备裁剪结果(第一个总是完整图像)
    crops = [image_pil.copy()]

    # 添加检测到的区域(最多max_crops-1个)
    for box in boxes[:max_crops - 1]:
        x1, y1, x2, y2 = map(int, box)
        crops.append(image_pil.crop((x1, y1, x2, y2)))

    # 不足时用完整图像填充
    while len(crops) < max_crops:
        crops.append(image_pil.copy())

    return crops[:max_crops]


def box_area(boxes):
    """计算检测框面积"""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def read_image():
    image_list = {}
    #替换为你的实际路径
    file_list = ["D:/fake_news/qiaojiao/dataset_weibo/nonrumor_images/", "D:/fake_news/qiaojiao/dataset_weibo/rumor_images/"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')
    yolo_model = load_yolov8()

    for path in file_list:
        for i, filename in enumerate(os.listdir(path)):
            try:
                im = Image.open(path + filename)
                # 使用YOLOv8检测和裁剪
                crops = detect_and_crop(im, yolo_model)
                # 处理每个裁剪区域
                processed_crops = []
                for crop in crops:
                    processed_crop = preprocess(crop).unsqueeze(0).to(device)
                    processed_crops.append(processed_crop)
                # 沿批次维度堆叠裁剪
                im_tensor = torch.mean(torch.cat(processed_crops, dim=0), dim=0) if len(processed_crops) > 0 else torch.zeros(3, 224, 224)
                image_list[filename.split('/')[-1].split(".")[0].lower()] = im_tensor
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    # print("image length " + str(len(image_list)))
    return image_list


class bert_data():
    def __init__(self, max_len, batch_size, num_workers=2):
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image = read_image()

    def load_data_train(self, path, text_only=False):
        self.data = pd.read_csv(path, encoding='utf-8')
        post = self.data
        ordered_image = []
        post_id = []
        image_id_list = []
        image_id = ""

        for i, id in enumerate(post['post_id']):
            for image_id in post.iloc[i]['image_id'].split('|'):
                image_id = image_id.split("/")[-1].split(".")[0]
                if image_id in self.image:
                    break

            if text_only or image_id in self.image:
                if not text_only:
                    image_name = image_id
                    image_id_list.append(image_name)
                    ordered_image.append(self.image[image_name])
                post_id.append(id)

        ordered_image = torch.tensor([item.cpu().detach().numpy() for item in ordered_image]).squeeze(1)
        print(ordered_image.size())
        with open('data/new_train_clip_crops_loader.pkl', 'wb') as file:
            pickle.dump(ordered_image, file)
        return 1

    def load_data_test(self, path, text_only=False):
        self.data = pd.read_csv(path, encoding='utf-8')
        post = self.data
        ordered_image = []
        post_id = []
        image_id_list = []
        image_id = ""

        for i, id in enumerate(post['post_id']):
            for image_id in post.iloc[i]['image_id'].split('|'):
                image_id = image_id.split("/")[-1].split(".")[0]
                if image_id in self.image:
                    break

            if text_only or image_id in self.image:
                if not text_only:
                    image_name = image_id
                    image_id_list.append(image_name)
                    ordered_image.append(self.image[image_name])
                post_id.append(id)

        ordered_image = torch.tensor([item.cpu().detach().numpy() for item in ordered_image]).squeeze(1)
        print(ordered_image.size())
        with open('data/new_test_clip_crops_loader.pkl', 'wb') as file:
            pickle.dump(ordered_image, file)
        return 1

    def load_data_val(self, path, text_only=False):
        self.data = pd.read_csv(path, encoding='utf-8')
        post = self.data
        ordered_image = []
        post_id = []
        image_id_list = []
        image_id = ""

        for i, id in enumerate(post['post_id']):
            for image_id in post.iloc[i]['image_id'].split('|'):
                image_id = image_id.split("/")[-1].split(".")[0]
                if image_id in self.image:
                    break

            if text_only or image_id in self.image:
                if not text_only:
                    image_name = image_id
                    image_id_list.append(image_name)
                    ordered_image.append(self.image[image_name])
                post_id.append(id)

        ordered_image = torch.tensor([item.cpu().detach().numpy() for item in ordered_image]).squeeze(1)
        print(ordered_image.size())
        with open('data/new_val_clip_crops_loader.pkl', 'wb') as file:
            pickle.dump(ordered_image, file)
        return 1


loader = bert_data(max_len=170, batch_size=64,num_workers=1)

val_loader = loader.load_data_val("data/csv/val_origin.csv", True)
test_loader = loader.load_data_test("data/csv/test_origin.csv", True)
train_loader = loader.load_data_train("data/csv/train_origin.csv", True)
