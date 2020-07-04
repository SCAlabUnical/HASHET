# HASHET
## HAshtag recommendation using Sentence-to-Hashtag Embedding Translation

The increasing use of microblogging platforms generates a huge amount of shared posts, leading to the need of effective
methods for categorization and search. In Twitter, hashtags are exploited by users for facilitating research and spread of trending
topics. However, choosing the correct hashtags is not very easy for users and tweets are often published without hashtags. To deal with
this issue we propose a new model, called HASHET,
aimed at suggesting a relevant set of hashtags for a given post. HASHET is based on two independent latent spaces for embedding
the text of a microblog post and the hashtags it contains. A mapping process based on a multilayer perceptron is then used for learning
a translation from the semantic features of the microblog text to the latent representation of its hashtags. HASHET has been applied to
a dataset of tweets related to the 2016 US presidential elections.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software:

```
- Python 3.6
```

### Installing
- Install requirements
```
pip install requirements.txt 
```
### Use
- Run methodology
```
python run.py
```

## Dataset

The dataset available in the `input/` folder is a sample of 100 tweets which has the sole purpose of showing the functioning of the methodology.
Each tweet is a json formatted string.

## Parameters
`constants.py` contains all the parameters used in the methodology. Changing them will influence the obtained results.
It is recommended to change `W2V_MINCOUNT` and `MINCOUNT` values for larger datasets.