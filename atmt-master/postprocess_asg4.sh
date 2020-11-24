infile=model_translations_nbest_k3_alpha09_no_diversity.txt
outfile=model_translations_nbest_k3_alpha09_no_diversity.out
lang=en

cat $infile | sed -r 's/(@@ )|(@@ ?$)//g' | perl moses_scripts/detruecase.perl | perl moses_scripts/detokenizer.perl -q -l $lang > $outfile
# cat $outfile | sacrebleu data_asg4/raw_data/test.en # computes bleuscore and n-gram precision
# important: here change reference path for sacrebleu to data_asg4/raw_data/test.en
# sacrebleu commented out for the nbest approach as this makes not sense and the test and outfile are not of the same dimensionsnning1096

