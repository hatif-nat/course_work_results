fast_align/build/fast_align -i trans_mbart_extra.txt -d -o -v > forward.align
fast_align/build/fast_align -i trans_mbart_extra.txt -d -o -v -r > reverse.align
fast_align/build/atools -i forward.align -j reverse.align -c grow-diag-final-and > trans_mbart_extra_align

fast_align/build/fast_align -i only_target_sen_trans_mbart_extra.txt -d -o -v > forward.align
fast_align/build/fast_align -i only_target_sen_trans_mbart_extra.txt -d -o -v -r > reverse.align
fast_align/build/atools -i forward.align -j reverse.align -c grow-diag-final-and > only_target_sen_trans_mbart_extra_align

awesome-align --data_file  trans_mbart.txt --output_file align_mbart_awesome --model_name_or_path model_without_co --no_cuda

awesome-align --data_file  only_target_sen_trans_mbart.txt --output_file only_target_sen_trans_mbart_awesome --model_name_or_path model_without_co
