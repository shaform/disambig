GLOVE = $(DISAMBIG_TOOL)/glove
WORD2VEC = $(DISAMBIG_TOOL)/word2vec
NLPPARSER = $(DISAMBIG_TOOL)/stanford-parser-full
LIBSVM_TRAIN = $(DISAMBIG_TOOL)/liblinear/train
LIBSVM_SCALE = $(DISAMBIG_TOOL)/libsvm/train-scale
CRFSUITE = $(DISAMBIG_TOOL)/crfsuite/frontend/crfsuite
TMP = /tmp

## -- data preprocessing -- ##

CLUE_CORPUS = $(DISAMBIG_BIG_DATA)/raw/uniqClueWeb4300W.no_cnnct.txt
CLUE_CORPUS_NP = $(DISAMBIG_BIG_DATA)/raw/NoPOSuniqClueWeb4300W.txt
CLUE_CORPUS_CNNCT = $(DISAMBIG_BIG_DATA)/raw/uniqClueWeb4300W.txt
LABELED_CLUE_CORPUS = $(DISAMBIG_BIG_DATA)/connective/4300W.labeled.txt
CNNCT_CORPUS = $(DISAMBIG_BIG_DATA)/connective/corpus.txt
CLUE_SENT_VECTOR = $(DISAMBIG_BIG_DATA)/connective/4300W.sent.vectors.txt
CLUE_WORD_VECTOR = $(DISAMBIG_BIG_DATA)/connective/4300W.word.vectors.txt

CDTB_RAW_GB_DIR = $(DISAMBIG_BIG_DATA)/raw/corpus
CDTB_RAW_DIR = $(DISAMBIG_BIG_DATA)/raw/corpus-utf8
LCORPUS_FILE = $(DISAMBIG_DATA)/raw_corpus/cdtb.txt
LINKAGE_FILE = $(DISAMBIG_DATA)/linkage/cdtb_linkage.txt
ARGUMENT_FILE = $(DISAMBIG_DATA)/linkage/cdtb_argument.txt
ARGUMENT_PREDICT_FILE = $(DISAMBIG_DATA)/linkage/cdtb_argument_predict.txt
LCORPUS_DEP_FILE = $(DISAMBIG_DATA)/raw_corpus/cdtb.dep.txt

GVOCAB_FILE = $(TMP)/vocab.txt
GCC_FILE = $(TMP)/cc.txt
GCCS_FILE = $(TMP)/ccs.txt
GGLOVE_VECTOR_FILE = $(DISAMBIG_BIG_DATA)/glove/4300W.vectors.txt
GGLOVE_NP_VECTOR_FILE = $(DISAMBIG_BIG_DATA)/glove/4300W.vectors.no_pos.txt

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

d_np_glove_prep:
	$(GLOVE)/vocab_count -min-count 5 -verbose 2 < $(CLUE_CORPUS_NP) > $(GVOCAB_FILE).np
	$(GLOVE)/cooccur -memory 20 -vocab-file $(GVOCAB_FILE).np -verbose 2 -window-size 15 < $(CLUE_CORPUS_NP) > $(GCC_FILE).np
	$(GLOVE)/shuffle -memory 20 -verbose 2 < $(GCC_FILE).np > $(GCCS_FILE).np

# create glove
d_np_glove:
	$(GLOVE)/glove -save-file $(TMP)/glove.np.vectors -threads 24 -input-file $(GCCS_FILE).np -iter 15 -x-max 10 -vector-size 400 -binary 0 -vocab-file $(GVOCAB_FILE).np -verbose 2 
	mv $(TMP)/glove.np.vectors.txt $(GGLOVE_NP_VECTOR_FILE)

# convert cdtb to utf-8
l_convert_cdtb_to_utf8:
	python3 $(DISAMBIG_PRG)/utility/common/gb-to-utf8.py --input $(DISAMBIG_BIG_DATA)/raw/corpus/ --output $(DISAMBIG_BIG_DATA)/raw/corpus-utf8/

c_extract_ntu_connectives:
	python3 $(DISAMBIG_PRG)/utility/connective/select.py --corpus $(CLUE_CORPUS_CNNCT) --pairs $(DISAMBIG_BIG_DATA)/raw/sqldump/intra_connectives.txt --output $(NTU_CNNCT)

# label sentences for connective experiments
c_process_corpus:
	python3 $(DISAMBIG_PRG)/utility/connective/filter.py --corpus $(CLUE_CORPUS_CNNCT) --cnnct $(NTU_CNNCT) --output $(TMP)/4300W.labeled.txt
	cp $(TMP)/4300W.labeled.txt $(LABELED_CLUE_CORPUS)

# construct sentence & word vectors for clueweb
c_sent_word2vec:
	shuf $(LABELED_CLUE_CORPUS) > $(TMP)/4300W.labeled.txt
	time $(WORD2VEC)/word2vec -train $(TMP)/4300W.labeled.txt -output $(TMP)/4300W.sent.vectors.txt -cbow 0 -size 400 -window 10 -negative 5 -hs 1 -sample 1e-3 -threads 24 -binary 0 -iter 20 -min-count 1 -sentence-vectors 1

# extract desired vectors to files
c_sent_extract:
	grep '^@@SSENT' $(TMP)/4300W.sent.vectors.txt | sort > $(CLUE_SENT_VECTOR)
	cat $(TMP)/4300W.sent.vectors.txt | sed '1d' | grep -v '^@@S' > $(CLUE_WORD_VECTOR)

c_preprocess:
	grep '^@@SSENT' $(LABELED_CLUE_CORPUS) > $(TMP)/4300W.filtered.txt
	python3 $(DISAMBIG_PRG)/utility/connective/preprocess.py --input $(TMP)/4300W.filtered.txt --output $(CNNCT_CORPUS)

d_convert_cdtb_encoding:
	python3 $(DISAMBIG_PRG)/utility/common/gb_to_utf8.py --input $(CDTB_RAW_GB_DIR) --output $(CDTB_RAW_DIR)

# extract cdtb linkages
d_extract_cdtb_linkage:
	PYTHONPATH=$(DISAMBIG_PRG)/linkage python3 $(DISAMBIG_PRG)/utility/linkage/cdtb_extractor.py --input $(CDTB_RAW_DIR) --output $(TMP)/corpus.raw.txt --connective $(TMP)/linkage.raw.txt --arg $(TMP)/linkage.arg.txt
	PYTHONPATH=$(DISAMBIG_PRG)/linkage python3 $(DISAMBIG_PRG)/utility/linkage/cdtb_align.py --corpus $(LCORPUS_FILE) --linkage_output $(TMP)/linkage.tmp.txt --argument_output $(ARGUMENT_FILE) --connective $(TMP)/linkage.raw.txt --arg $(TMP)/linkage.arg.txt --argument_text $(TMP)/arg_text.txt
	uniq $(TMP)/linkage.tmp.txt > $(LINKAGE_FILE)

# generate parser structure
# first stripped labels
d_parse_cdtb:
	$(DISAMBIG_TOOL)/stanford-parser-full/lexparser-lang.sh Chinese 500 edu/stanford/nlp/models/lexparser/chinesePCFG.ser.gz parsed $(DISAMBIG_DATA)/cdtb.stripped.txt
# then concate labels again
d_process_cdtb_parsed:
	python3 $(DISAMBIG_PRG)/utility/common/align_parser.py --corpus $(LCORPUS_FILE) --parsed $(DISAMBIG_DATA)/cdtb.stripped.txt.parsed.500.stp --output $(TMP)/cdtb.parsed.txt

# generate folds
d_make_folds:
	python3 $(DISAMBIG_PRG)/utility/linkage/split_folds.py --folds 10 --corpus $(LCORPUS_FILE) --linkage $(LINKAGE_FILE) --output $(TMP)/folds.txt

# filter vectors
d_filter_vectors:
	python3 $(DISAMBIG_PRG)/utility/linkage/filter_vectors.py --vectors $(GGLOVE_VECTOR_FILE) --corpus_pos $(LCORPUS_POS_FILE) --output $(LVECTOR_FILE)

# filter word2vec vectors
d_filter_w2v_vectors:
	python3 $(DISAMBIG_PRG)/utility/linkage/filter_vectors.py --vectors $(CLUE_WORD_VECTOR) --corpus_pos $(LCORPUS_POS_FILE) --output $(LW2V_VECTOR_FILE)

d_dep_parse:
	PYTHONPATH=$(DISAMBIG_PRG)/linkage python3 $(DISAMBIG_PRG)/utility/linkage/dep_parse.py --corpus $(LCORPUS_FILE) --input $(TMP)/NONE --label $(TMP)/labels.txt --output $(TMP)/tmp.txt preprocess
	java -Xmx"60g" -cp "$(CLASSPATH)":"$(NLPPARSER)/*" edu.stanford.nlp.parser.lexparser.LexicalizedParser -maxLength 600 -tLPP edu.stanford.nlp.parser.lexparser.ChineseTreebankParserParams -chineseFactored -encoding UTF-8 -tokenized -sentences newline -escaper edu.stanford.nlp.trees.international.pennchinese.ChineseEscaper -writeOutputFiles -outputFormat "typedDependencies" -outputFormatOptions "removeTopBracket,includePunctuationDependencies" -loadFromSerializedFile edu/stanford/nlp/models/lexparser/chineseFactored.ser.gz $(TMP)/tmp.txt
	PYTHONPATH=$(DISAMBIG_PRG)/linkage python3 $(DISAMBIG_PRG)/utility/linkage/dep_parse.py --corpus $(LCORPUS_FILE) --label $(TMP)/labels.txt --input $(TMP)/tmp.txt.stp --output $(LCORPUS_DEP_FILE) postprocess

## -- connective experiments -- ##

c_experiment:
	PYTHONPATH=$(DISAMBIG_PRG)/linkage python3 $(DISAMBIG_PRG)/connective/experiment.py --corpus $(CNNCT_CORPUS) --output $(TMP)/test.txt # --train $(LIBSVM_TRAIN) --scale $(LIBSVM_SCALE)

## -- linkage experiments -- ##

LCNNCT_FILE = $(DISAMBIG_DATA)/connectives-simple/cdt_cdtb.txt
LCORPUS_POS_FILE = $(DISAMBIG_DATA)/raw_corpus/cdtb.pos.txt
LCORPUS_PARSE_FILE = $(DISAMBIG_DATA)/parsed/cdtb.parsed.txt
LVECTOR_FILE = $(DISAMBIG_DATA)/linkage/cdtb_vectors.txt
LW2V_VECTOR_FILE = $(DISAMBIG_DATA)/linkage/cdtb_w2v_vectors.txt
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
	python3 $(DISAMBIG_PRG)/linkage/extract_word_features.py --tag $(LCNNCT_FILE) --corpus $(LCORPUS_FILE) --corpus_pos $(LCORPUS_POS_FILE) --corpus_parse $(LCORPUS_PARSE_FILE) --linkage $(LINKAGE_FILE) --vector $(LVECTOR_FILE) --output $(LWORD_FEATURE_FILE) --output_ambig $(LWORD_AMBIG_FILE) #--select GLOVE

# 2. train word probabilities
l_train_word_probs:
	python3 $(DISAMBIG_PRG)/linkage/train_word_probs.py --word_ambig $(LWORD_AMBIG_FILE) --word_features $(LWORD_FEATURE_FILE) --folds $(LFOLDS_FILE) --output $(LWORD_PROB_FILE) --check_accuracy

# 3. extract features for each linkage
l_extract_linkage_features:
	python3 $(DISAMBIG_PRG)/linkage/extract_linkage_features.py --tag $(LCNNCT_FILE) --corpus $(LCORPUS_FILE) --corpus_pos $(LCORPUS_POS_FILE) --corpus_parse $(LCORPUS_PARSE_FILE) --linkage $(LINKAGE_FILE) --vector $(LVECTOR_FILE) --folds $(LFOLDS_FILE) --output $(LLINKAGE_FEATURE_FILE) --perfect_output $(LLINKAGE_PFEATURE_FILE) #--select GLOVE  #--check_accuracy

# 4. train linkage probabilities
l_train_linkage_probs:
	python3 $(DISAMBIG_PRG)/linkage/train_linkage_probs.py --linkage_features $(LLINKAGE_FEATURE_FILE) --linkage $(LINKAGE_FILE) --folds $(LFOLDS_FILE) --output $(LLINKAGE_PROB_FILE) --check_accuracy

# 4.5. train perfect linkage probabilities
l_train_perfect_linkage_probs:
	python3 $(DISAMBIG_PRG)/linkage/train_linkage_probs.py --linkage_features $(LLINKAGE_PFEATURE_FILE) --linkage $(LINKAGE_FILE) --folds $(LFOLDS_FILE) --output $(LLINKAGE_PPROB_FILE) --check_accuracy

# 5. run the experiments
l_experiment:
	python3 $(DISAMBIG_PRG)/linkage/experiment.py --tag $(LCNNCT_FILE) --word_ambig $(LWORD_AMBIG_FILE) --folds $(LFOLDS_FILE) --corpus $(LCORPUS_FILE) --corpus_pos $(LCORPUS_POS_FILE) --corpus_parse $(LCORPUS_PARSE_FILE) --word_probs $(LWORD_PROB_FILE) --linkage $(LINKAGE_FILE) --linkage_probs $(LLINKAGE_PROB_FILE) --linkage_features $(LLINKAGE_FEATURE_FILE) --arg_output $(ARGUMENT_PREDICT_FILE) --check_accuracy #--threshold 6.0

# 5.5 run the perfect experiments
l_perfect_experiment:
	python3 $(DISAMBIG_PRG)/linkage/experiment.py --tag $(LCNNCT_FILE) --word_ambig $(LWORD_AMBIG_FILE) --folds $(LFOLDS_FILE) --corpus $(LCORPUS_FILE) --corpus_pos $(LCORPUS_POS_FILE) --corpus_parse $(LCORPUS_PARSE_FILE) --word_probs $(LWORD_PROB_FILE) --linkage $(LINKAGE_FILE) --linkage_probs $(LLINKAGE_PPROB_FILE) --perfect --check_accuracy

# 6. run sense experiment
l_sense_experiment:
	python3 $(DISAMBIG_PRG)/linkage/sense_experiment.py --linkage_features $(LLINKAGE_PFEATURE_FILE) --linkage $(LINKAGE_FILE)

# 7. run argument experiment
l_arg_experiment:
	python3 $(DISAMBIG_PRG)/linkage/arg_experiment.py --folds $(LFOLDS_FILE) --corpus $(LCORPUS_FILE) --corpus_pos $(LCORPUS_POS_FILE) --corpus_parse $(LCORPUS_PARSE_FILE) --corpus_dep $(LCORPUS_DEP_FILE) --argument_test $(ARGUMENT_PREDICT_FILE) --argument $(ARGUMENT_FILE) --crfsuite $(CRFSUITE) --train $(TMP)/crftrain.txt --test $(TMP)/crftest.txt --model $(TMP)/crf.model --log $(TMP)/error_analysis.txt --hierarchy_adjust #--keep_boundary

# 8. run perfect argument experiment
l_perfect_arg_experiment:
	python3 $(DISAMBIG_PRG)/linkage/arg_experiment.py --folds $(LFOLDS_FILE) --corpus $(LCORPUS_FILE) --corpus_pos $(LCORPUS_POS_FILE) --corpus_parse $(LCORPUS_PARSE_FILE) --corpus_dep $(LCORPUS_DEP_FILE) --argument_test $(ARGUMENT_FILE) --argument $(ARGUMENT_FILE) --crfsuite $(CRFSUITE) --train $(TMP)/crftrain.txt --test $(TMP)/crftest.txt --model $(TMP)/crf.model --log $(TMP)/error_analysis.txt --hierarchy_adjust #--keep_boundary

l_statistics:
	PYTHONPATH=$(DISAMBIG_PRG)/linkage python3 $(DISAMBIG_PRG)/utility/linkage/statistics.py --tag $(LCNNCT_FILE) --corpus $(LCORPUS_FILE) --linkage $(LINKAGE_FILE)

## -- test experiments -- ##

t_arg_crf_train:
	for i in 0 1 2 3 4 5 6 7 8 9 ; do $(CRFSUITE) learn -m $(TMP)/crf.model.$$i $(TMP)/crftrain.txt.$$i ; done
t_arg_crf_test:
	for i in 0 1 2 3 4 5 6 7 8 9 ; do $(CRFSUITE) tag -qt -m $(TMP)/crf.model.$$i $(TMP)/crftest.txt.$$i | grep Instance ; done
