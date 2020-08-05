# HIERARCHICAL MULTI-SCALE ATTENTION FOR SEMANTIC SEGMENTATION
多尺度感知的改良 by Nvidia 2020

## Introduction
同一張圖片中，可能因為辨識物體大小的差異影響到model辨識的精準度，Liang-Chieh Chen於2015年提出過輸入不同分辨率的影像來使model學習各自對應的物體之attention權重。
![](https://i.imgur.com/S6hLnWV.jpg)


### 多尺度實現
- 目的: 對於一張圖片進行segmentation
- 方法: 將原圖片在一個範圍內進行up sampling 跟 down sampling，會得到一組大小尺寸不同的圖片，送入segmentation model中之後各自得到其結果後再恢復為原本之大小。
- 發現: 大尺度的圖得到小物體的語意分割效果較佳，小尺度的圖片得到大物體(風格)的分割效果較好。==一般採用平均的方式得到最後的結果。==
- 問題: 使用平均值組合時會遇到結合的問題，降低精準度，因此本文提出使用attention來取代average的方法來使網路選擇attention高的部分進行選用。

## Hierarchical multi-scale attention架構
![](https://i.imgur.com/dnQ4NdQ.jpg)
### backbone
- study: ResNet-50
- SOTA result: HRNet-OCR
### Single Scale vs. multi scale 
![](https://i.imgur.com/GH8i2h8.jpg)


## Result
### Attention可視化
![](https://i.imgur.com/Mxiq3ve.jpg)
### mIOU Result
![](https://i.imgur.com/l9DHclU.jpg)
### Auto labeling
![](https://i.imgur.com/WJ7J1mq.jpg)

## 可看paper方向:
- HRNet
- OCR






