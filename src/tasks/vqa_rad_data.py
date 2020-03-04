# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from param import args
from utils import load_obj_tsv
from nltk.translate.bleu_score import corpus_bleu

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000

# The path to data and image features.
#VQA_RAD_DATA_ROOT = 'data/vqa_rad/VQA_RAD_Images'
#VQA_RAD_IMGFEAT_ROOT = 'data/vqa_rad/vqa_rad_imgfeat'
VQA_RAD_DATA_ROOT = 'data/combined_data/'
VQA_RAD_IMGFEAT_ROOT = 'data/combined_data/split'
SPLIT2NAME = {
    'train': 'train',
    'valid': 'val',
    'minival': 'val',
    'test': 'test',
}


class VQARADDataset:
    """
    A VQA data example in json file:
        {
            "answer_type": "other",
            "img_id": "COCO_train2014_000000458752",
            "label": {
                "net": 1
            },
            "question_id": 458752000,
            "question_type": "what is this",
            "sent": "What is this photo taken looking through?"
        }
    """
    def __init__(self, splits: str):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open(
                os.path.join(VQA_RAD_DATA_ROOT,
                             "split/%s.json" % SPLIT2NAME[split]))))
        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

        # Answers
        self.ans2label = json.load(open(os.path.join(VQA_RAD_DATA_ROOT,
                             "split/trainval_ans2label.json")))
        self.label2ans = json.load(open(os.path.join(VQA_RAD_DATA_ROOT,
                             "split/trainval_label2ans.json")))
        assert len(self.ans2label) == len(self.label2ans)

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""
class VQARADTorchDataset(Dataset):
    def __init__(self, dataset: VQARADDataset):
        super().__init__()
        self.raw_dataset = dataset

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = None

        # Loading detection features to img_data
        img_data = []
        for split in dataset.splits:
            # Minival is 5K images in MS COCO, which is used in evaluating VQA/LXMERT-pre-training.
            # It is saved as the top 5K features in val2014_***.tsv
            load_topk = 5000 if (split == 'minival' and topk is None) else topk
            img_data.extend(load_obj_tsv(
                os.path.join(VQA_RAD_IMGFEAT_ROOT, '%s_obj36.tsv' % (SPLIT2NAME[split])),
                topk=load_topk))

        # Convert img list to dict
        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data:
            if datum['img_id'] in self.imgid2img:
                self.data.append(datum)
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()
        assert obj_num == len(boxes) == len(feats)

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                target[self.raw_dataset.ans2label[ans]] = score
            return ques_id, feats, boxes, ques, target
        else:
            return ques_id, feats, boxes, ques


class VQARADEvaluator:
    def __init__(self, dataset: VQARADDataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans in label:
                score += label[ans]
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }

        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'question_id': ques_id,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)

    @staticmethod
    def calculate_bleu(predictions_with_truth, category=None):
        hypothesis_list = []
        references_list = []
        if category == "closed":
            weights = (1.0, 0.0, 0.0, 0.0)
        else:
            weights = (0.5, 0.5, 0.0, 0.0)
        for p in predictions_with_truth:
            if category is not None:
                if p['answer_type'].strip().lower() == category:
                    hypothesis_list.append(str(p['answer']).strip().lower().split())
                    references_list.append([p['prediction'].strip().lower().split()])
            else:
                hypothesis_list.append(str(p['answer']).strip().lower().split())
                references_list.append(
                    [p['prediction'].strip().lower().split()])

        return corpus_bleu(references_list, hypothesis_list, weights=weights)

    @staticmethod
    def evaluate_bleu(ground_truth_json, predictions_json,
                      output_json,
                      rephrase_idx_shift=10000,
                      category=('closed', 'open')):
        with open(predictions_json) as f:
            predictions = json.load(f)

        with open(ground_truth_json) as f:
            ground_truth = json.load(f)

        # get the ground truth
        ground_truth_dict = {}
        for g in ground_truth:
            ground_truth_dict[g['qid']] = g

        # predictions with ground truth
        # Also calcuate accuracy for closed answer type
        count = 0
        correct_count = 0
        predictions_with_truth = []
        for p in predictions:
            qid = int(p['question_id'])
            rephrase = False
            if qid > 9999:
                qid -= rephrase_idx_shift
                rephrase = True
            d = ground_truth_dict[qid].copy()
            d['prediction'] = str(p['answer'])
            d['rephrased_question'] = rephrase
            predictions_with_truth.append(d)


            if d['answer_type'].strip().lower() == category[0]:
                count += 1
                correct_count += int(d['prediction'].strip().lower() == d['answer'].strip().lower())

        accuracy = None
        if count != 0:
            accuracy = (float(correct_count) / count) * 100

        all, closed, opened = VQARADEvaluator.calculate_bleu(predictions_with_truth,
                                                             category=None), \
        VQARADEvaluator.calculate_bleu(predictions_with_truth,
                                       category=category[0]), \
                              VQARADEvaluator.calculate_bleu(
                                  predictions_with_truth,
                                  category=category[1])


        # write out the combined predictions json
        with open(os.path.join(output_json),'w') as g:
            json.dump(predictions_with_truth, g, sort_keys=True, indent=4)

        return {'all': all, category[0]: closed, category[1]: opened}, accuracy









