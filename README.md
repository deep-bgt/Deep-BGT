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

       1. Please download corpora by command " wget https://gitlab.com/parseme/sharedtask-data/-/archive/master/sharedtask-data-master.zip "
          Unzip the downloaded file
          Locate it into CICLing_42/input/corpora
       2. All word embeddings are available in the following links:
       
            [BG](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bg.300.vec.gz),
            [DE](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.vec.gz),
            [EL](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.el.300.vec.gz),
            [EN](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz),
            [ES](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.es.300.vec.gz),
            [EU](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.eu.300.vec.gz),
            [FA](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fa.300.vec.gz),
            [FR](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.vec.gz),
            [HE](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.he.300.vec.gz),
            [HI](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.hi.300.vec.gz),
            [HR](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.hr.300.vec.gz),
            [HU](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.hu.300.vec.gz),
            [IT](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.it.300.vec.gz),
            [LT](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.lt.300.vec.gz),
            [PL](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pl.300.vec.gz),
            [PT](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pt.300.vec.gz),
            [RO](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ro.300.vec.gz),
            [SL](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sl.300.vec.gz),
            [TR](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.tr.300.vec.gz).
          Download word embeddings by command " wget language_link "
          Locate them into CICLing_42/input/embeddings
       3. Language codes are Bulgarian (BG), German (DE), Greek (EL), English (EN), Spanish (ES), Basque (EU), Farsi (FA), French (FR),
          Hebrew (HE), Hindu (HI), Crotian (HR), Hungarian (HU), Italian (IT), Lithuanian (LT),
           Polish (PL), Portuguese (PT), Romanian (RO), Slovenian (SL), and Turkish (TR).

Setup with virtual environment (Python 3):
-  python3 -m venv CICLing_42_venv
   source CICLing_42_venv/bin/activate
- Install the requirements:
   CICLing_42_venv/bin/pip3 install -r requirements.txt

If everything works well, you can run the example usage described below.


### Example Usage:
- The following guide show an example usage of the model for English with bigappy-unicrossy tagging scheme.
- Instructions
      
      1. Download word embeddings: " wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/cc.en.300.vec.gz "
         Locate it into CICLing_42/input/embeddings
      2. Download the PARSEME Corpora: " wget https://gitlab.com/parseme/sharedtask-data/-/archive/master/sharedtask-data-master.zip "
         Unzip the downloaded file
         Locate it into CICLing_42/input/corpora
      3. Change directory to the location of the source code which is CICLing_42
      4. Run the instructions in "Setup with virtual environment (Python 3)"
      5. Run the command to train the model: python3 Runner.py -l EN -t gappy-crossy
         If you want to try the model with another configuration, change language code after -l, and tag after -t
         Languages: BG, DE, EL, EN, ES, EU, FA, FR, HE, HI HR, HU, LT, IT, PL, PT, RO, SL, TR
         Tags: IOB, gappy-1, gappy-crossy
      6. Open the file in CICLing_42/eval.cmd, copy and run the command in this file to evaluate the accuracy
      7. The results will be in CICLing_42/output/EN/eval.txt
