---
language: "en"
thumbnail: "https://pbs.twimg.com/profile_images/1092721745994440704/d6R-AHzj_400x400.jpg"
tags:
- propaganda
- bert
license: "MIT"
datasets:
-
metrics:
-
---

Propaganda Techniques Analysis BERT
----

This model is a BERT based model to make predictions of propaganda techniques in
news articles in English. It was introduced in
[this paper](https://propaganda.qcri.org/papers/EMNLP_2019__Fine_Grained_Propaganda_Detection.pdf).


## Model description

Please find propaganda definition here:
https://propaganda.qcri.org/annotations/definitions.html


## How to use

```python
>>> from transformers import BertTokenizerFast
>>> from .model import BertForTokenAndSequenceJointClassification
>>>
>>> tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
>>> model = BertForTokenAndSequenceJointClassification.from_pretrained(
>>>     "QCRI/PropagandaTechniquesAnalysis-en-BERT",
>>>     revision="v0.1.0",
>>> )
>>> 
>>> inputs = tokenizer.encode_plus("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)
>>> sequence_class_index = torch.argmax(outputs.sequence_logits, dim=-1)
>>> sequence_class = model.sequence_tags[sequence_class_index[0]]
>>> token_class_index = torch.argmax(outputs.token_logits, dim=-1)
>>> tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0][1:-1])
>>> tags = [model.token_tags[i] for i in token_class_index[0].tolist()[1:-1]]
```
