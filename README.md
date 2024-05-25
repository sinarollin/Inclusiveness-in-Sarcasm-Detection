# Inclusiveness in Sarcasm Detection
> Sandra Frey, Céline Hirsch, Sina Röllin

### Abstract 💡


### Research Questions 🔍


### Installation ⚙️

To install all the libraries needed for this project it is best to create a virtual environment. To do so, open a terminal and run the following command:

`conda create -n DLproject python=3.11`

Then activate the virtual environment and navigate to this repository:

`conda activate DLproject`
`cd path/to/Inclusiveness-in-Sarcasm-Detection`

You can then install all of the necessary packages at once, using the following command:

`pip install -r requirements.txt`


### Data 📈

We used the publicly available MUStARD dataset for multi-modal sarcasm detection, which contains text, audio and video data from multiple TV sitcoms. The data includes the *utterance of interest*, *context*, *speaker*, and *sarcasm label*. The data was manually annotated to include the *gender* of the speaker, and all utterances for which the speakers’ gender was unclear were removed. Three datasets were created, one containing only female utterances, one containing only male utterances and the final one containing both. These datasets were balanced to ensure the inclusion of the same amount of sarcastic and non-sarcastic utterances for each gender by data augmentation. 

### Methods 📚

#### Audio Extraction


#### Text Data Augmentation
The data augmentation for the text utterances was performed using synonym replacement. Synonyms for verbs, adjectives or pronouns were extracted from WordNet based on a similarity of $> 0.6$ 

#### Audio Data Augmentation
