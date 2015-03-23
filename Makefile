GLOVE = $(DISAMBIG_TOOL)/glove
TMP = /tmp

## -- data preprocessing -- ##

CLUE_CORPUS=$(DISAMBIG_BIG_DATA)/raw/uniqClueWeb4300W.no_cnnct.txt
GVOCAB_FILE=$(TMP)/vocab.txt
GCC_FILE=$(TMP)/cc.txt
GCCS_FILE=$(TMP)/ccs.txt
GGLOVE_VECTOR_FILE=$(DISAMBIG_BIG_DATA)/glove/4300W.vectors.txt

# prepare files to create glove output
l_glove_prep:
	$(GLOVE)/vocab_count -min-count 5 -verbose 2 < $(CLUE_CORPUS) > $(GVOCAB_FILE)
	$(GLOVE)/cooccur -memory 20 -vocab-file $(GVOCAB_FILE) -verbose 2 -window-size 15 < $(CLUE_CORPUS) > $(GCC_FILE)
	$(GLOVE)/shuffle -memory 20 -verbose 2 < $(GCC_FILE) > $(GCCS_FILE)

# create glove
l_glove:
	$(GLOVE)/glove -save-file $(TMP)/glove.vectors -threads 24 -input-file $(GCCS_FILE) -iter 15 -x-max 10 -vector-size 400 -binary 0 -vocab-file $(GVOCAB_FILE) -verbose 2 
	mv $(TMP)/glove.vectors.txt $(GGLOVE_VECTOR_FILE)

## -- disambig experiments -- ##

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

# a. generate some stats
l_count:
	python3 $(DISAMBIG_PRG)/linkage/count_all_linkage.py $(LCORPUS_FILE) $(LCNNCT_FILE) $(LINKAGE_FILE)

# b. generate parser structure
# first stripped labels
l_parse_cdtb:
	$(DISAMBIG_TOOL)/stanford-parser-full/lexparser-lang.sh Chinese 500 edu/stanford/nlp/models/lexparser/chinesePCFG.ser.gz parsed $(DISAMBIG_DATA)/cdtb.stripped.txt
# then concate labels again
l_process_cdtb_parsed:
	python3 $(DISAMBIG_PRG)/utility/align_parser.py --corpus $(LCORPUS_FILE) --parsed $(DISAMBIG_DATA)/cdtb.stripped.txt.parsed.500.stp --output $(TMP)/cdtb.parsed.txt

# c. generate folds
l_make_folds:
	python3 $(DISAMBIG_PRG)/utility/linkage/split_folds.py --folds 10 --corpus $(LCORPUS_FILE) --linkage $(LINKAGE_FILE) --output $(TMP)/folds.txt

# d. filter vectors
l_filter_vectors:
	python3 $(DISAMBIG_PRG)/utility/linkage/filter_vectors.py --vectors $(GGLOVE_VECTOR_FILE) --corpus_pos $(LCORPUS_POS_FILE) --output $(LVECTOR_FILE)
