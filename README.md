# Interpretable Multimodal Hate-Speech

The following repository contains source code to train Gradient Boosted Trees, Deep Neural Networks for the detection of multimodal hate speech detection.


## Get started (Requirements and Setup)
Python version >= 3.6

``` bash
​```  clone the repository
git clone git@github.com:TIBHannover/Interpretable-Multimodal-Hate-Speech.git
cd Interpretable-Multimodal-Hate-Speech
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


## Hate words and 365 classes

``` bash
​``` 
resources/hatewords.txt
resources/hateword_365_classes.jsonl
```



TODOs

- [ ] Upload the extracted features (CLIP, image & text) to Zenodo
- [ ] Refactor the repository for training CLIP-DNN, DNN, XGBoost

