infile=model_translations_k3_alpha08new.txt
outfile=model_translations_k3_alpha08new.out
lang=en

cat $infile | sed -r 's/(@@ )|(@@ ?$)//g' | perl moses_scripts/detruecase.perl | perl moses_scripts/detokenizer.perl -q -l $lang > $outfile
cat $outfile | sacrebleu data_asg4/raw_data/test.en # computes bleuscore and n-gram precision
# important: here change reference path for sacrebleu to data_asg4/raw_data/test.en
