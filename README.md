# Verbal Multiword Expression (VMWE) Identification

This repository contains an implementation of the **Bidirectional LSTM-CRF** model for VMWE identification and a novel tagging scheme **bigappy-unicrossy** for representation of overlaps in sequence labeling tasks.

## PARSEME shared task on automatic identification of verbal MWEs - edition 1.1

The PARSEME shared task 2018 on automatic identification of verbal MWEs (VMWE) covers 20 languages. The corpora provided are in cupt format and include annotations of VMWEs consisting of categories. The categories of VMWEs are light verb constructions with two subcategories (LVC.full and LVC.cause), verbal idioms (VID), inherently reflexive verbs (IRV), verb-particle constructions with two subcategories (VPC.full and VPC.semi), multi-verb constructions (MVC), inherently adpositional verbs (IAV) and inherently clitic verbs (LS.ICV).

The corpora has been published at [LINDAT/CLARIN](https://lindat.mff.cuni.cz/repository/xmlui/handle/11372/LRT-2842).

We are the annotators of Turkish corpus. The paper describing our annotation process is [Turkish Verbal Multiword Expressions Corpus](https://ieeexplore.ieee.org/document/8404583).

## Bidirectional LSTM-CRF Model for Verbal Multiword Expression Identification

The Deep-BGT system participated to the PARSEME shared task 2018. Our system is language-independent and uses the bidirectional Long Short-Term Memory model with a Conditional Random Field layer on top (bidirectional LSTM-CRF).

The architecture is decribed in our paper: [Deep-BGT at PARSEME Shared Task 2018: Bidirectional LSTM-CRF Model for Verbal Multiword Expression Identification](https://aclanthology.info/papers/W18-4927/w18-4927)

To the best of our knowledge, this paper is the first one that employs the bidirectional LSTM-CRF model for VMWE identification. 

The gappy 1-level tagging scheme is used. Our system was evaluated on 10 languages in the open track and it was ranked the second in terms of the general ranking metric. [(Shared Task Results)](http://multiword.sourceforge.net/PHITE.php?sitesig=CONF&page=CONF_04_LAW-MWE-CxG_2018___lb__COLING__rb__&subpage=CONF_50_Shared_task_results)

Later on, we evaluated our system on 19 languages. The results will be published in Springer's Lecture Notes in Computer Science (LNCS), CICLing 2019.

## A Novel Tagging Scheme: bigappy-unicrossy

We introduce a new tagging scheme called **bigappy-unicrossy** to rise to the challenge of overlapping MWEs.

The **bigappy-unicrossy** tagging scheme is compared with the *IOB2* and the *gappy 1-level* tagging schemes in VMWE identification task (PARSEME shared task) using our previous system *Deep-BGT*. It is evaluated our system on 19 languages. The results will be published in Springer's Lecture Notes in Computer Science (LNCS), CICLing 2019.

## Citation

If you make use of our implementation regarding the deep learning architecture, please cite the following paper: [Deep-BGT at PARSEME Shared Task 2018: Bidirectional LSTM-CRF Model for Verbal Multiword Expression Identification](https://aclanthology.info/papers/W18-4927/w18-4927)

If you make use of our implementation regarding the tagging scheme, please cite the following paper: Representing Overlaps in Sequence Labeling Tasks with a Novel Tagging Scheme: bigappy-unicrossy (will be published in Springer's Lecture Notes in Computer Science (LNCS), CICLing 2019).

## Implementation

### Requirements
- Python 3.6
- Keras 2.2.4 with Tensorflow 1.12.0, and keras-contrib==2.0.8
- We cannot guarantee that the code works with different versions for Keras / Tensorflow.
- We cannot provide the data used in the experiments in this code repository, because we have no right to distribute the corpora provided by PARSEME Shared Task Edition 1.1 .

       1. Please download corpora from https://gitlab.com/parseme/sharedtask-data
          Unzip the downloaded file
          Locate it into input/corpora
       2. All word embeddings are available in the https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md
          Download word embeddings and locate them into input/embeddings
       3. Language codes are Bulgarian (BG), German (DE), Greek (EL), English (EN), Spanish (ES), Basque (EU), Farsi (FA), French (FR),
          Hebrew (HE), Hindu (HI), Crotian (HR), Hungarian (HU), Italian (IT), Lithuanian (LT),
           Polish (PL), Portuguese (PT), Romanian (RO), Slovenian (SL), and Turkish (TR).

#### Setup with virtual environment (Python 3):
-  python3 -m venv my_venv

   source my_venv/bin/activate
- Install the requirements:
   pip3 install -r requirements.txt

If everything works well, you can run the example usage described below.


### Example Usage:
- The following guide show an example usage of the model for English with bigappy-unicrossy tagging scheme.
- Instructions
      
      1. Change directory to the location of the source code
      2. Run the instructions in "Setup with virtual environment (Python 3)"
      3. Run the command to train the model: python3 src/Runner.py -l EN -t gappy-crossy
         If you want to try the model with another configuration, change language code after -l, and tag after -t
         Languages: BG, DE, EL, EN, ES, EU, FA, FR, HE, HI HR, HU, LT, IT, PL, PT, RO, SL, TR
         Tags: IOB, gappy-1, gappy-crossy
