The data file ('humanPredictions.json') contains DR (discourse referent) predictions crowdsourced using Amazon Mturk. 
This data needs to be used in conjunction with InScript Corpus, which is included in the corresponding directory "data"

The file is in json format and has the following format:

{docID:{tokenID:{trueDRLabel :[dr1,dr2,....dr20]}} , 

where dr1,dr2,....are DR predictions by humans.  

The data files are in (tab separated) column format, where the columns have the following meaning:

1 = id
2 = word
3 = headVerbID_dependencyRelation
4 = verbLemma
5 = POS
6 = coref_participant_label