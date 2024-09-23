# HASHET
## HAshtag recommendation using Sentence-to-Hashtag Embedding Translation
<div style="text-align: justify">
The growing use of microblogging platforms is generating a huge amount of posts that need effective methods to be classified and searched. In Twitter and other social media platforms, hashtags are exploited by users to facilitate the search, categorization and spread of posts. Choosing the appropriate hashtags for a post is not always easy for users, and therefore posts are often published without hashtags or with hashtags not well defined. To deal with this issue, we propose a new model, called HASHET (HAshtag recommendation using Sentence-to-Hashtag Embedding Translation), aimed at suggesting a relevant set of hashtags for a given post. HASHET is based on two independent latent spaces for embedding the text of a post and the hashtags it contains. A mapping process based on a multilayer perceptron is then used for learning a translation from the semantic features of the text to the latent representation of its hashtags. We evaluated the effectiveness of two language representation models for sentence embedding and tested different search strategies for semantic expansion, finding out that the combined use of BERT (Bidirectional Encoder Representation from Transformer) and a global expansion strategy leads to the best recommendation results.
HASHET has been evaluated on two real-world case studies related to the 2016 United States presidential election and COVID-19 pandemic.
The results reveal the effectiveness of HASHET in predicting one or more correct hashtags, with an average F-score up to 0.82 and a recommendation hit-rate up to 0.92.
Our approach has been compared to the most relevant techniques used in the literature (generative models, unsupervised models and attention-based supervised models) by achieving up to 15% improvement in F-score for the hashtag recommendation task and 9% for the topic discovery task.

## How to cite
R. Cantini, F. Marozzo, G. Bruno, P. Trunfio, "Learning sentence-to-hashtags semantic mapping for hashtag recommendation on microblogs". ACM Transactions on Knowledge Discovery from Data, vol. 16, n. 2, pp. 1-26, 2022.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and 
testing purposes.

### Prerequisites

```
- Python 3.7
```

### Installing
- Install requirements
```
pip install requirements.txt 
python -m spacy download en_core_web_lg
```
### Use
- Run the HASHET model
```
python run.py
```

## Dataset

The dataset available in the `input/` folder is a sample of 100 tweets which has the sole purpose of showing 
the functioning of the methodology. Each tweet is a json formatted string.

The real datasets on which HASHET has been validated are in the `used_dataset` folder.
In accordance with Twitter API Terms, only Tweet IDs are provided as part of this datasets. 
To recollect tweets based on the list of Tweet IDs contained in these datasets you will need to use tweet 
'rehydration' programs.

The resulting json line for each tweet after rehydration must have this format:
```
{
   "id":"id",
   "text":"tweet text",
   "date":"date",
   "user":{
      "id":"user_id",
      "name":"",
      "screenName":"",
      "location":"",
      "lang":"en",
      "description":""
   },
   "location":{
      "latitude":0.0,
      "longitude":0.0
   },
   "isRetweet":false,
   "retweets":0,
   "favoutites":0,
   "inReplyToStatusId":-1,
   "inReplyToUserId":-1,
   "hashtags":[
      "hashtag"
   ],
   "lang":"lang",
   "place":{      
   }
}
```

## Parameters
`constants.py` contains all the parameters used in the methodology. Changing them will influence the obtained results.
It is recommended to change `W2V_MINCOUNT` and `MINCOUNT` values for larger datasets.

</div>
