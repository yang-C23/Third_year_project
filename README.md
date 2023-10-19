# MedTem: Medications and Treatments Mining and Temporal Relation Modelling using NLP

<img src="https://github.com/HECTA-UoM/MedTem/blob/main/MedTem-logo-v1.png" width="100">



The main task of this project is to automatically determine whether the current treatment is used during the patient's hospital stay based on the doctor's notes on the patient and the given treatment entities.

## Installation and Configuration

The project is completed using Jupyter notebook and can be run on Jupyter notebook or Colab.
Python version: 3.9.16 (main, Dec  7 2022, 01:11:51) 
CUDA version: 11.6
Transformers version: 4.27.3

## Usage

1. It can implement few-shot learning and the size of the dataset can be set when collecting data.
2. You can set different plm, template and verbalizer when using prompt-based learning.
3. You can set different plm for fine-tuning method.

## Dataset

I used Evaluating temporal relations in clinical text: 2012 i2b2 Challenge as my dataset

## Cite our work:

[HealTAC2023 presentation](http://healtex.org/healtac-2023/programme/)

Yang Cui, Lifeng Han, Goran Nenadic. 2023. MedTem2.0: Prompt-based Temporal Classification of Treatment Events from Discharge Summaries. Accepted to ACL2023: SRW. [paper](https://aclanthology.org/2023.acl-srw.27/) [poster](MedTem2_poster_Portrait_4ACL.pdf)

MedTem-1: @misc{tu2023extraction, title={Extraction of Medication and Temporal Relation from Clinical Text using Neural Language Models}, author={Hangyu Tu and Lifeng Han and Goran Nenadic}, year={2023}, eprint={2310.02229}, archivePrefix={arXiv}, url={https://arxiv.org/abs/2310.02229} primaryClass={cs.CL} }

Hangyu Tu. 2022. Extraction of Temporal Information from Clinical Free Text. MSc thesis. Uni Manchester. MSc Advisor: [Prof Goran Nenadic](https://research.manchester.ac.uk/en/persons/gnenadic) and [Dr Lifeng Han](https://research.manchester.ac.uk/en/persons/lifeng.han) | [thesis-download](https://www.researchgate.net/publication/369453637_Extraction_of_Temporal_Information_from_Clinical_Free_Text)