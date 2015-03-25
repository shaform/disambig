GLOVE = $(DISAMBIG_TOOL)/glove
WORD2VEC = $(DISAMBIG_TOOL)/word2vec
TMP = /tmp

## -- data preprocessing -- ##

CLUE_CORPUS = $(DISAMBIG_BIG_DATA)/raw/uniqClueWeb4300W.no_cnnct.txt
CLUE_CORPUS_CNNCT = $(DISAMBIG_BIG_DATA)/raw/uniqClueWeb4300W.txt
LABELED_CLUE_CORPUS = $(DISAMBIG_BIG_DATA)/connective/4300W.labeled.txt
GVOCAB_FILE = $(TMP)/vocab.txt
GCC_FILE = $(TMP)/cc.txt
GCCS_FILE = $(TMP)/ccs.txt
GGLOVE_VECTOR_FILE = $(DISAMBIG_BIG_DATA)/glove/4300W.vectors.txt
NTU_CNNCT = $(DISAMBIG_DATA)/connective/ntu_connective.txt

# prepare files to create glove output
d_glove_prep:
	$(GLOVE)/vocab_count -min-count 5 -verbose 2 < $(CLUE_CORPUS) > $(GVOCAB_FILE)
	$(GLOVE)/cooccur -memory 20 -vocab-file $(GVOCAB_FILE) -verbose 2 -window-size 15 < $(CLUE_CORPUS) > $(GCC_FILE)
	$(GLOVE)/shuffle -memory 20 -verbose 2 < $(GCC_FILE) > $(GCCS_FILE)

# create glove
d_glove:
	$(GLOVE)/glove -save-file $(TMP)/glove.vectors -threads 24 -input-file $(GCCS_FILE) -iter 15 -x-max 10 -vector-size 400 -binary 0 -vocab-file $(GVOCAB_FILE) -verbose 2 
	mv $(TMP)/glove.vectors.txt $(GGLOVE_VECTOR_FILE)

# convert cdtb to utf-8
l_convert_cdtb_to_utf8:
	python3 $(DISAMBIG_PRG)/utility/common/gb-to-utf8.py --input $(DISAMBIG_BIG_DATA)/raw/corpus/ --output $(DISAMBIG_BIG_DATA)/raw/corpus-utf8/

# get connectives from experiments
# cat $(DISAMBIG_BIG_DATA)/raw/sqldump/intra_connectives.txt | cut -f3,4,7 > $(TMP)/ntu_cnnct.txt
# $(NTU_CNNCT)
c_extract_ntu_connectives:
	python3 $(DISAMBIG_PRG)/utility/connective/select.py --corpus $(CLUE_CORPUS_CNNCT) --pairs $(DISAMBIG_BIG_DATA)/raw/sqldump/intra_connectives.txt --output $(NTU_CNNCT)

c_process_corpus:
	python3 $(DISAMBIG_PRG)/utility/connective/filter.py --corpus $(CLUE_CORPUS_CNNCT) --cnnct $(NTU_CNNCT) --output $(TMP)/4300W.labeled.txt
	cp $(TMP)/4300W.labeled.txt $(LABELED_CLUE_CORPUS)

c_sent_word2vec:
	shuf $(LABELED_CLUE_CORPUS) > $(TMP)/4300W.labeled.txt
	time $(WORD2VEC)/word2vec -train $(TMP)/4300W.labeled.txt -output $(TMP)/4300W.sent.vectors.txt -cbow 0 -size 400 -window 10 -negative 5 -hs 1 -sample 1e-3 -threads 24 -binary 0 -iter 20 -min-count 1 -sentence-vectors 1

## -- connective experiments -- ##

## -- linkage experiments -- ##

LINKAGE_FILE = $(DISAMBIG_DATA)/linkage/cdtb_linkage.txt
LCNNCT_FILE = $(DISAMBIG_DATA)/connectives-simple/cdt_cdtb.txt
LCORPUS_FILE = $(DISAMBIG_DATA)/raw_corpus/cdtb.txt
LCORPUS_POS_FILE = $(DISAMBIG_DATA)/raw_corpus/cdtb.pos.txt
LCORPUS_PARSE_FILE = $(DISAMBIG_DATA)/parsed/cdtb.parsed.txt
LVECTOR_FILE = $(DISAMBIG_DATA)/linkage/cdtb_vectors.txt
LWORD_FEATURE_FILE = $(DISAMBIG_DATA)/linkage/cdtb_word_features.txt
LWORD_PROB_FILE = $(DISAMBIG_DATA)/linkage/cdtb_word_probs.txt
LWORD_AMBIG_FILE = $(DISAMBIG_DATA)/linkage/cdtb_word_ambig.txt
LLINKAGE_PROB_FILE = $(DISAMBIG_DATA)/linkage/cdtb_linkage_probs.txt
LLINKAGE_PPROB_FILE = $(DISAMBIG_DATA)/linkage/cdtb_linkage_perfect_probs.txt
LLINKAGE_FEATURE_FILE = $(DISAMBIG_DATA)/linkage/cdtb_linkage_features.txt
LLINKAGE_PFEATURE_FILE = $(DISAMBIG_DATA)/linkage/cdtb_linkage_perfect_features.txt
LFOLDS_FILE = $(DISAMBIG_DATA)/linkage/cdtb_10folds.txt
L10FOLDS_FILE = $(DISAMBIG_DATA)/linkage/cdtb_10folds.txt

# 1. extract features for each word
l_extract_word_features:
	python3 $(DISAMBIG_PRG)/linkage/extract_word_features.py --tag $(LCNNCT_FILE) --corpus $(LCORPUS_FILE) --corpus_pos $(LCORPUS_POS_FILE) --corpus_parse $(LCORPUS_PARSE_FILE) --linkage $(LINKAGE_FILE) --vector $(LVECTOR_FILE) --output $(LWORD_FEATURE_FILE) --output_ambig $(LWORD_AMBIG_FILE)

# 2. train word probabilities
l_train_word_probs:
	python3 $(DISAMBIG_PRG)/linkage/train_word_probs.py --word_ambig $(LWORD_AMBIG_FILE) --word_features $(LWORD_FEATURE_FILE) --folds $(LFOLDS_FILE) --output $(LWORD_PROB_FILE) --check_accuracy

# 3. extract features for each linkage
l_extract_linkage_features:
	python3 $(DISAMBIG_PRG)/linkage/extract_linkage_features.py --tag $(LCNNCT_FILE) --corpus $(LCORPUS_FILE) --corpus_pos $(LCORPUS_POS_FILE) --corpus_parse $(LCORPUS_PARSE_FILE) --linkage $(LINKAGE_FILE) --vector $(LVECTOR_FILE) --word_probs $(LWORD_PROB_FILE) --folds $(LFOLDS_FILE) --output $(LLINKAGE_FEATURE_FILE) --perfect_output $(LLINKAGE_PFEATURE_FILE) #--check_accuracy

# 4. train linkage probabilities
l_train_linkage_probs:
	python3 $(DISAMBIG_PRG)/linkage/train_linkage_probs.py --linkage_features $(LLINKAGE_FEATURE_FILE) --linkage $(LINKAGE_FILE) --folds $(LFOLDS_FILE) --output $(LLINKAGE_PROB_FILE) --check_accuracy

# 4.5. train perfect linkage probabilities
l_train_perfect_linkage_probs:
	python3 $(DISAMBIG_PRG)/linkage/train_linkage_probs.py --linkage_features $(LLINKAGE_PFEATURE_FILE) --linkage $(LINKAGE_FILE) --folds $(LFOLDS_FILE) --output $(LLINKAGE_PPROB_FILE) --check_accuracy

# 5. run the experiments
l_experiment:
	python3 $(DISAMBIG_PRG)/linkage/experiment.py --tag $(LCNNCT_FILE) --word_ambig $(LWORD_AMBIG_FILE) --folds $(LFOLDS_FILE) --corpus $(LCORPUS_FILE) --corpus_pos $(LCORPUS_POS_FILE) --corpus_parse $(LCORPUS_PARSE_FILE) --word_probs $(LWORD_PROB_FILE) --linkage $(LINKAGE_FILE) --linkage_probs $(LLINKAGE_PROB_FILE) --check_accuracy

# 5.5 run the perfect experiments
l_perfect_experiment:
	python3 $(DISAMBIG_PRG)/linkage/experiment.py --tag $(LCNNCT_FILE) --word_ambig $(LWORD_AMBIG_FILE) --folds $(LFOLDS_FILE) --corpus $(LCORPUS_FILE) --corpus_pos $(LCORPUS_POS_FILE) --corpus_parse $(LCORPUS_PARSE_FILE) --word_probs $(LWORD_PROB_FILE) --linkage $(LINKAGE_FILE) --linkage_probs $(LLINKAGE_PPROB_FILE) --check_accuracy

# a. generate parser structure
# first stripped labels
l_parse_cdtb:
	$(DISAMBIG_TOOL)/stanford-parser-full/lexparser-lang.sh Chinese 500 edu/stanford/nlp/models/lexparser/chinesePCFG.ser.gz parsed $(DISAMBIG_DATA)/cdtb.stripped.txt
# then concate labels again
l_process_cdtb_parsed:
	python3 $(DISAMBIG_PRG)/utility/common/align_parser.py --corpus $(LCORPUS_FILE) --parsed $(DISAMBIG_DATA)/cdtb.stripped.txt.parsed.500.stp --output $(TMP)/cdtb.parsed.txt

# b. generate folds
l_make_folds:
	python3 $(DISAMBIG_PRG)/utility/linkage/split_folds.py --folds 10 --corpus $(LCORPUS_FILE) --linkage $(LINKAGE_FILE) --output $(TMP)/folds.txt

# c. filter vectors
l_filter_vectors:
	python3 $(DISAMBIG_PRG)/utility/linkage/filter_vectors.py --vectors $(GGLOVE_VECTOR_FILE) --corpus_pos $(LCORPUS_POS_FILE) --output $(LVECTOR_FILE)
