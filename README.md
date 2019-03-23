VMWE Identification

Requirements
- Python 3.6
- Keras 2.2.4 with Tensorflow 1.12.0, and keras-contrib==2.0.8
- We cannot guarantee that the code works with different versions for Keras / Tensorflow.
- We cannot provide the data used in the experiments in this code repository, because we have no right to distribute the corpora provided by PARSEME Shared Task Edition 1.1 .
       1. Please download corpora by command " wget https://gitlab.com/parseme/sharedtask-data/-/archive/master/sharedtask-data-master.zip "
          Unzip the downloaded file
          Locate it into CICLing_42/input/corpora
       2. All word embeddings are available in the following links:
            BG: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bg.300.vec.gz
            DE: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.vec.gz
            EL: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.el.300.vec.gz
            EN: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz
            ES: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.es.300.vec.gz
            EU: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.eu.300.vec.gz
            FA: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fa.300.vec.gz
            FR: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.vec.gz
            HE: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.he.300.vec.gz
            HI: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.hi.300.vec.gz
            HR: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.hr.300.vec.gz
            HU: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.hu.300.vec.gz
            IT: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.it.300.vec.gz
            LT: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.lt.300.vec.gz
            PL: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pl.300.vec.gz
            PT: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pt.300.vec.gz
            RO: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ro.300.vec.gz
            SL: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sl.300.vec.gz
            TR: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.tr.300.vec.gz
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

Example Usage:
- The following guide show an example usage of the model for English with bigappy-unicrossy tagging scheme.
- Running all experiments for 19 languages with the three different tagging schemes -IOB, gappy-1-level, and bigappy-unicrossy- takes at least one week.
- Since English is one the smallest corpus, we choose this example to show our new tagging scheme.
- Running this experiment will take approximately a few hours.
- Instructions
      1. Download word embeddings: " wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/cc.en.300.vec.gz "
         Locate it into CICLing_42/input/embeddings
      2. Download the PARSEME Corpora: " wget https://gitlab.com/parseme/sharedtask-data/-/archive/master/sharedtask-data-master.zip "
         Unzip the downloaded file
         Locate it into CICLing_42/input/corpora
      3. Change directory to the location of the source code which is CICLing_42
      4. Run the instructions in "Setup with virtual environment (Python 3)"
      5. Run the command to train the model: python3 src/Runner.py -l EN -t gappy-crossy
         If you want to try the model with another configuration, change language code after -l, and tag after -t
         Languages: BG, DE, EL, EN, ES, EU, FA, FR, HE, HI HR, HU, LT, IT, PL, PT, RO, SL, TR
         Tags: IOB, gappy-1, gappy-crossy
      6. Open the file in CICLing_42/eval.cmd, copy and run the command in this file to evaluate the accuracy
      7. The results will be in CICLing_42/output/EN/eval.txt
