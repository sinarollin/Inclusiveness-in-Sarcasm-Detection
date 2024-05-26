# Inclusiveness in Sarcasm Detection
> Authors: Sandra Frey, Céline Hirsch, Sina Röllin

### Abstract 💡

Sarcasm usage and style can vary significantly between men and women. Previous studies on sarcasm detection using deep learning models have largely overlooked gender considerations. This study explores the subtle differences in how men and women employ sarcasm, highlighting the need for gender inclusiveness in training deep learning models for sarcasm detection. By training text and audio models on single-gender and mixed datasets, the study evaluates the influence of gender-specific nuances in sarcasm usage on model performance. Results indicate that gender plays a crucial role in model performance, and emphasize the importance of gender-balanced deep learning models to enhance the generalisability of future research.

### Installation ⚙️

To install all the libraries needed for this project it is best to create a virtual environment. To do so, open a terminal and run the following command:

`conda create -n DLproject python=3.11`

Then activate the virtual environment and navigate to this repository:

`conda activate DLproject`
`cd path/to/Inclusiveness-in-Sarcasm-Detection`

You can then install all of the necessary packages at once, using the following command:

`pip install -r requirements.txt`


### Data 📈

We used the publicly available [MUStARD dataset for multi-modal sarcasm detection](https://github.com/soujanyaporia/MUStARD), which contains text, audio and video data from multiple TV sitcoms. The data includes the *utterance of interest*, *context*, *speaker*, and *sarcasm label*. The data was manually annotated to include the *gender* of the speaker, and all utterances for which the speakers’ gender was unclear were removed. Three datasets were created, one containing only female utterances, one containing only male utterances and the final one containing both. These datasets were balanced to ensure the inclusion of the same amount of sarcastic and non-sarcastic utterances for each gender by data augmentation. 

### Methods 📚

#### Text Data Augmentation
This step was done in the file `text-data-preparation.ipynb`. The data augmentation for the text utterances was performed using synonym replacement. Synonyms for verbs, adjectives or pronouns were extracted from WordNet based on a similarity of $> 0.6$. 

#### Audio Data Augmentation and Extraction
The audio was extracted from the videos provided in the original dataset and was then transformed to a waveform format in order to produce embeddings for the model input. These were done in the file `audio_embeddings_data_preparation_1.ipynb` and `audio_embeddings_data_preparation_2.ipynb`. The data augmentation for the audio samples was performed in two different ways: by lowering the pitch of chosen samples and by introducing a certain level of noise to the audio waveforms.

#### Text Model
This step was done in the file `text-model-training.ipynb`. For the text sarcasm detection we used a tinyBERT model. Three models were trained this way: one female data only, one on male data only and one on mixed data. 

#### Audio Model
This step was done in the file `audio_embeddings_training.ipynb`. For the audio sarcasm detection we used a Wav2Vec2 model. Three models were trained this way: one female data only, one on male data only and one on mixed data. 

#### Model Evaluation
This step was done in the files `text-model-evaluation.ipynb` and `audio_embeddings_training.ipynb`. To test all our models we assessed their performance on three different datasets: female data only, male data only and mixed data. We used accuracy, loss and F1 score. 


#### Note for Audio Data:
As the spectrograms as well as the embeddings are too large even in compressed form, they can be found on the following site: https://drive.google.com/drive/folders/1-sEyHQU6l0RqNHKS0toeIE2pDIAtDssL?usp=sharing.
