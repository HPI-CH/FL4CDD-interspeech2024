## A Comparative Analysis of Federated Learning for Speech-Based Cognitive Decline Detection

authors: Stefan Kalabakov, Monica Gonzalez Machorro, Florian Eyben, Björn Schuller and Bert Arnrich

e-mail addresses: Stefan.Kalabakov@hpi.de / Monica.Gonzalez@tum.de


Abstract:

Speech-based machine learning models that can distinguish between a healthy cognitive state and different stages of cognitive decline would enable a more appropriate and timely treatment of patients. However, their development is often hampered by data scarcity. Federated Learning (FL) is a potential solution that could enable entities with limited voice recordings to collectively build effective models. Motivated by this, we compare centralised, local, and federated learning for building speech-based models to discern Alzheimer’s Disease, Mild Cognitive Impairment, and a healthy state. For a more realistic evaluation, we use three independently collected datasets to simulate healthcare institutions employing these strategies. Our initial analysis shows that FL may not be the best solution in every scenario, as performance improvements are not guaranteed even with small amounts of available data, and further research is needed to determine the conditions under which it is beneficial.

##  Installation

```
$ virtualenv --python=python3.9 fl/
$ source fl/bin/activate
(fl) $ pip install -r requirements.txt
(fl) $ main.py
```