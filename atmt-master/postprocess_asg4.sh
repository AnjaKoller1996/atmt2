infile=model_translations_nbest.txt
outfile=model_translations_nbest.out
lang=en

cat $infile | sed -r 's/(@@ )|(@@ ?$)//g' | perl moses_scripts/detruecase.perl | perl moses_scripts/detokenizer.perl -q -l $lang > $outfile
cat $outfile | sacrebleu data_asg4/raw_data/test_nbest.en # computes bleuscore and n-gram precision
# important: here change reference path for sacrebleu to data_asg4/raw_data/test.en
