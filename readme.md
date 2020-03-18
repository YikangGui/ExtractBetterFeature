# README

The topic of project is related to zero shot learning. In the image classification task, zero shot learning means that the category in the test data set will not appear in the train data set.

The data source is coming from Alibaba Tian Chi competition. The data set contains three parts: train data set, test data set and word embedding. Train data set contains 365 classes and 151124 images and the corresponded label. Test data set contains 65 classes and 8025 images. Word embedding data set contains word embedding for all categories in both train and test data set. There is no overlap between train dataset and test dataset with respect to category.

In the zero shot learning, the whole task will be divided into three components: extracting image feature model, mapping between image feature and class attribute or class word embedding model and classification by KNN model.

There are two section:
- `Classification_Model`: to extract model
- `ZeroShot_Model`: achieve map between wordembedding and image feature and predict image label
NOTE: if you need the data, please download from https://tianchi.aliyun.com/competition/entrance/231677/information?lang=en-us

## Image Features Extraction
![r](assets/model1.png)

## Mapping
![r](assets/model2.png)

## KNN Model
![r](assets/model3.png)
