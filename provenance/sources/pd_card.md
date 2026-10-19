---
license: cc-by-4.0
dataset_info:
  features:
  - name: image_id
    dtype: int64
  - name: image
    dtype: image
  - name: width
    dtype: int64
  - name: height
    dtype: int64
  - name: objects
    struct:
    - name: id
      sequence: int64
    - name: area
      sequence: int64
    - name: bbox
      sequence:
        sequence: float32
    - name: category
      sequence: string
  splits:
  - name: train
    num_bytes: 905619617.284
    num_examples: 2342
  - name: test
    num_bytes: 73503583
    num_examples: 236
  download_size: 991825068
  dataset_size: 979123200.284
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
task_categories:
- object-detection
---


This Dataset is created from processing the files from this GitHub repository : PlantDoc-Object-Detection-Dataset

@inproceedings{10.1145/3371158.3371196,
author = {Singh, Davinder and Jain, Naman and Jain, Pranjali and Kayal, Pratik and Kumawat, Sudhakar and Batra, Nipun},
title = {PlantDoc: A Dataset for Visual Plant Disease Detection},
year = {2020},
isbn = {9781450377386},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3371158.3371196},
doi = {10.1145/3371158.3371196},
booktitle = {Proceedings of the 7th ACM IKDD CoDS and 25th COMAD},
pages = {249–253},
numpages = {5},
keywords = {Deep Learning, Object Detection, Image Classification},
location = {Hyderabad, India},
series = {CoDS COMAD 2020}
}
