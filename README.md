# Radiology VQA

### Code that is different from LXMERT
1. VQA-RAD specific data processing is in ``data/vqa_rad``
1. Image Clef 2019 specific data processing is in ``data/imageclef2019``
1. Scripts to combine both the data ``data/combined_data``
1. Model, data loader and training code is in ``src/vqa_rad_model.py, src/vqa_rad_data.py, src/vqa_rad.py``
1. Scripts to trigger training and testing``run/vqa_rad_finetune.bash, run/vqa_rad_test.bash``
1. Test predication evaluation is in ``run/vqa_rad_data.py``


## Finetuning using Lxmert





