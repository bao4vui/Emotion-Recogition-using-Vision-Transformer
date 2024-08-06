# Emotion-Recognition-using-Vision-Transformer
This repository contains an implementation of a Vision Transformer (ViT) for the FER2013 dataset, described in [paper](https://arxiv.org/abs/2010.11929).
## How to use my code 
Với mã nguồn này, bạn có thể: 
* Chạy hệ thống nhận diện cảm xúc con người 
* Kiểm thử dữ liệu với phần dữ liệu test của FER2013 hoặc dữ liệu riêng của bạn
## Requirements:
* python 3.11
* torch 2.3.0
* opencv-python 
* transformers
* ultralytics
* numpy
## Datasets:
|Datasets|Classes|Train images|Validation images|Test images|
|--------|-------|------------|-----------------|-----------|
|FER2013|7|28709|3589|3589|

Tải bộ dữ liệu FER2013 vào thư mục **dataset** qua đường link dưới đây:
[FER2013](https://www.kaggle.com/datasets/deadskull7/fer2013 "Fer2013")
* Dataset gốc: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
* Link bài báo dataset gốc: https://arxiv.org/pdf/1307.0414.pdf
## ViT Trained models
Các bạn có thể tìm được tất cả các phiên bản mô hình của ViT theo đường dẫn sau (sẽ cập nhật đường link sau): [ViT_Trained_model]()
## Training

## Testing

## Experiments 

## Results 

## Running live video 
Để có thể chạy chương trình nhận diện cảm xúc, trước hết, bạn cần phải chắc chắn rằng đã chứa mô hình sử dụng trong thư mục chứa mô hình, ví dụ:
* Nếu muốn dùng mô hình nhẹ, tải file model về thư mục, optimizer.pt, gigido.pt về thư mục **FER_VIT_model**.
* Nếu muốn dùng mô hình nặng hơn, tải file model về thư mục, optimizer.pt, gigido.pt về thư mục **FER_VIT_large_model**.
  
Sau khi đã hoàn tất tải mô hình bạn muốn vào thư mục tương ứng để thực hiện bài toán nhận diện cảm xúc, bạn hãy nhập dòng lệnh sau:
> python main.py
Lúc này, đợi cho máy tính chuẩn bị xong, ta sẽ có được kết quả hiển thị chương trình như hình sau: 
.
.
.
.