---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: label
    dtype:
      class_label:
        names:
          '0': Apple__Apple_scab
          '1': Apple__Black_rot
          '2': Apple__Cedar_apple_rust
          '3': Apple___healthy
          '4': Blueberry___healthy
          '5': Cherry_(including_sour)___Powdery_mildew
          '6': Cherry_(including_sour)___healthy
          '7': Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot
          '8': Corn_(maize)___Common_rust_
          '9': Corn_(maize)___Northern_Leaf_Blight
          '10': Corn_(maize)___healthy
          '11': Grape___Black_rot
          '12': Grape___Esca_(Black_Measles)
          '13': Grape___Leaf_blight_(Isariopsis_Leaf_Spot)
          '14': Grape___healthy
  splits:
  - name: train
    num_bytes: 183540951.55
    num_examples: 11322
  - name: test
    num_bytes: 45176971.724
    num_examples: 2916
  download_size: 219374135
  dataset_size: 228717923.27400002
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
---
