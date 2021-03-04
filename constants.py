
GUSE = "GUSE"
BERT = "BERT"
# choose encoder
SENTENCE_ENCODING_MODEL = GUSE

USA = "usa"
COVID = "covid"
# choose dataset
DATASET = USA

# input and savings folder
if DATASET == USA:
    INPUT_PATH = "input/"
    SAVE_FOLDER = "save_files/"
else:
    INPUT_PATH = "input_covid/"
    SAVE_FOLDER = "save_files_covid/"

W2V_INPUT = SAVE_FOLDER + "sentences.txt"
W2V_SAVE_FILE_NAME = SAVE_FOLDER + "w2v_emb.model"

# Word2Vec mincount parameter (suggested value for larger datasets: 5)
W2V_MINCOUNT = 5
# hashtag mincount in corpus (suggested value for large datasets: 10 or more)
MINCOUNT = 10

LATENT_SPACE_DIM = 150
WINDOW_SIZE = 5

# name of the input text file for the model
TRAIN_TEST_INPUT = SAVE_FOLDER + 'model_sentences.txt'

# skip hashtag cleaning before sentence embedding parameter
SKIP_HASHTAG_REMOVING = True

TRAIN_CORPUS = SAVE_FOLDER + "train_corpus.txt"
TEST_CORPUS = SAVE_FOLDER + "test_corpus.txt"

# Google Universal Sentence Encoder path
GUSE_PATH = "https://tfhub.dev/google/universal-sentence-encoder-large/5"

PERC_TEST = 0.25

H_REMOVING_DICT = SAVE_FOLDER + 'h_removing_dict.pkl'

MINMAX_SCALER_FILENAME = SAVE_FOLDER + 'minmax_scaler.pkl'

STD_SCALER_FILENAME = SAVE_FOLDER + 'std_scaler.pkl'

if SENTENCE_ENCODING_MODEL == GUSE:
    TL_MODEL_JSON_FILE_NAME = SAVE_FOLDER + "TLmodel.json"
    TL_MODEL_WEIGHTS_FILE_NAME = SAVE_FOLDER + "TLweights.h5"
    FT_MODEL_JSON_FILE_NAME = SAVE_FOLDER + "FTmodel.json"
    FT_MODEL_WEIGHTS_FILE_NAME = SAVE_FOLDER + "FTweights.h5"
else:
    TL_MODEL_JSON_FILE_NAME = SAVE_FOLDER + "BERT_TLmodel.json"
    TL_MODEL_WEIGHTS_FILE_NAME = SAVE_FOLDER + "BERT_TLweights.h5"
    FT_MODEL_JSON_FILE_NAME = SAVE_FOLDER + "BERT_FTmodel.json"
    FT_MODEL_WEIGHTS_FILE_NAME = SAVE_FOLDER + "BERT_FTweights.h5"

MODEL_JSON_FILE_NAME = FT_MODEL_JSON_FILE_NAME
MODEL_WEIGHTS_FILE_NAME = FT_MODEL_WEIGHTS_FILE_NAME

GLOBAL_EXPANSION = "global"
LOCAL_EXPANSION = "local"

# choose expansion strategy
EXPANSION_STRATEGY = GLOBAL_EXPANSION

MAX_EXPANSION_ITERATIONS = 6
# --------------------------

if DATASET == USA:
    TOPICS = [
        ["#clintokaine16", "#democrats", "#dems", "#dnc", "#dumpfortrump", "#factcheck",
         "#hillary16", "#hillary2016", "#hillarysupporter", "#hrc",
         "#imwithher", "#lasttimetrumppaidtaxes", "#nevertrump", "#ohhillyes", "#p2", "#strongertogether",
         "#trumptape", "#uniteblu", "#notmypresident"],
        ["#americafirst", "#benghazi", "#crookedhillary", "#draintheswamp", "#lockherup", "#maga3x", "#maga",
         "#makeamericagreatagain", "#neverhillary", "#podestaemails", "#projectveritas", "#riggedetection",
         "#tcot", "#trump2016", "#trumppence16", "#trumptrain", "#voterfraud", "#votetrump", "#gop",
         "#republicans", "#wakeupamerica", "#lockherup", "#hillarysemail", "#weinergate"]
    ]
else:
    TOPICS = [
        ['#covid19', '#coronavirus', '#covid_19', '#news', '#corona', '#uk', '#usa', '#india', '#coronaviruspandemic',
         '#travel', '#coronavirusstrain', '#breaking', '#covid20', '#coronavirusupdate', '#breakingnews',
         '#unitedkingdom', '#pakistan', '#southafrica', '#world', '#coronavirusupdates'],
        ['#california', '#florida', '#lasvegas', '#vegas', '#losangeles', '#protests', '#newyork', '#texas', '#miami',
         '#2019ncov', '#sanfrancisco', '#hawaii', '#arizona', '#ohio', '#tampa', '#pennsylvania', '#tampabay',
         '#houston', '#tennessee', '#oregon'],
        ['#christmas', '#merrychristmas', '#christmas2020', '#christmaseve', '#quarantine', '#happyholidays',
         '#covidchristmas', '#merrychristmas2020', '#2020', '#merryxmas', '#love', '#christmaseve2020', '#education',
         '#santa', '#mentalhealth', '#family', '#newyear', '#holidays', '#music', '#xmas2020'],
        ['#wonderwoman1984', '#ww84', '#thursdaymorning', '#starwars', '#wifematerial', '#cashapp', '#joinin',
         '#lightsforlouis', '#happybirthdaylouistomlinson', '#asuu', '#aobigdealblowout', '#quaidday', '#wizkid',
         '#davido', '#kogipostpandemic', '#ausvind', '#wednesdaythought', '#soulmovie', '#christmaslovebyjimin',
         '#indiawithpfi'],
        ['#wearamask', '#stayhome', '#staysafe', '#stayathome', '#socialdistancing', '#maskup', '#washyourhands',
         '#ppe',
         '#masks', '#stayhomestaysafe', '#mask', '#facemasks', '#socialdistance', '#facemask', '#stopthespread',
         '#stayhealthy', '#flagmask', '#besafe', '#protection', '#flattenthecurve'],
        ['#pandemic', '#workfromhome', '#jobs', '#business', '#wfh', '#remotejobs', '#money', '#remotework',
         '#workfromhomejobs', '#virtualassistant', '#onlinejobs', '#onlinejob', '#digitalnomad', '#bitcoin', '#nomad',
         '#economy', '#marketing', '#technology', '#makemoneyonline', '#finance'],
        ['#vaccine', '#health', '#covidvaccine', '#vaccines', '#healthcare', '#sarscov2', '#vaccination',
         '#frontlineheroes', '#virus', '#nhs', '#science', '#pfizer', '#covid19vaccine', '#coronavaccine', '#moderna',
         '#healthcareheroes', '#hospitals', '#publichealth', '#vaccineswork', '#vaccinessavelives'],
        ['#lockdown', '#covid19uk', '#boxingday', '#tier4', '#london', '#manchester', '#coronavirusuk', '#lockdown3',
         '#londonlockdown', '#uklockdown', '#ireland', '#lockdown2', '#coviduk', '#tier4lockdown', '#tier3',
         '#stormbella', '#wales', '#restrictions', '#justsaying', '#christmasiscancelled'],
        ['#canada', '#cdnpoli', '#onpoli', '#bcpoli', '#toronto', '#ontariolockdown', '#ontario', '#vancouver',
         '#covid19ab', '#ottawa', '#covid19ontario', '#ableg', '#calgary', '#covid19bc', '#covidcanada', '#edmonton',
         '#montreal', '#abpoli', '#bced', '#yeg'],
        ['#browns', '#nba', '#nfl', '#sports', '#football', '#rockets', '#worldjuniors', '#soundhound', '#fpl',
         '#nbatwitter', '#nbaxmas', '#mcfc', '#premierleague', '#jets', '#fantasyfootball', '#mancity', '#nufc',
         '#winning', '#lions', '#jamesharden'],
        ['#trump', '#republicans', '#trumpvirus', '#gop', '#maga', '#foxnews', '#plandemic', '#dopeydon',
         '#stimuluscheck', '#covidrelief', '#covidreliefpackage', '#greatreset', '#stimuluspackage',
         '#trumpisacompletefailure', '#america', '#georgia', '#donaldtrump', '#agenda21', '#americans', '#stimulus'],
        ['#brexit', '#brexitdeal', '#wearenotgoingaway', '#borisjohnson', '#borishasfailedthenation', '#boristheliar',
         '#kirbysigston', '#richmondnorthyorkshire', '#scotland', '#torycorruption', '#borisjohnsonmustgo',
         '#toryincompetence', '#bbc', '#dover', '#bbcnews', '#rejoineu', '#brexitreality', '#boris', '#toriesout',
         '#brexitdisaster'],
        ['#covid19nsw', '#7news', '#sydney', '#covid19aus', '#covidnsw', '#nswpol', '#9news', '#covid19vic',
         '#gladyscluster', '#nsw', '#gladysoutbreak', '#melbourne', '#springst', '#northernbeaches', '#victoria',
         '#strandedaussies', '#contacttracing', '#nswcovid19', '#scottyfrommarketing', '#istandwithdan']
    ]

# neural network
BATCH_SIZE = 32
# log levels
SILENT = 0
PROGRESS = 1
ONE_LINE_PER_EPOCH = 2
LOG_LEVEL = 1
PATIENCE = 3
MAX_EPOCHS = 50
# perc. of input layer neurons in the FC layer
