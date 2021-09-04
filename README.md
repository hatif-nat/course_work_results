# course_work_results
reproduse best results of course work.

results print on the screen and save in results.txt

To reproduse all results (translate + alignment) run:
```bash
./install_deps.sh
python3 download.py
python3 translate.py
python3 add_Tatoeba.py
./alignment.sh
python3 result.py
```
But mBart translator work very slow and crashes on low-power computers. You can reproduse result alignmrnt with pre-translated words:
```bash
./install_deps.sh
python3 download.py
python3 add_Tatoeba.py
./alignment.sh
python3 result.py
```

Used code, programm and resources from:  
https://github.com/UKPLab/EasyNMT  
https://github.com/clab/fast_align  
https://github.com/neulab/awesome-align  
https://russe.nlpub.org/2018/wsi/  
https://tatoeba.org/en/downloads  
