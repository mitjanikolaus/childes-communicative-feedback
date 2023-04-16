# CHILDES communicative feedback

This repository contains code accompanying the following paper:

**Communicative Feedback in Response to Children's Grammatical Errors** \
*In Proceedings of the 45th Annual Meeting of the Cognitive Science Society. (2023)* \
Mitja Nikolaus, Laurent Prévot and Abdellah Fourtassi

To reproduce the exact results from the paper, the
[cogsci_2023](https://github.com/mitjanikolaus/childes-communicative-feedback/tree/cogsci_2023) branch should be used.

## Python Environment

### Preprocessing and CF analyses

You can create an environment using the [environment.yml](environments/environment.yml) file (if you're not on Ubuntu, you can also
use the [environment_os_independent.yml](environments/environment_os_independent.yml) file:
```
conda env create --file environments/environment.yml
```
Additionally, we need to install [my fork of the pylangacq repo](https://github.com/mitjanikolaus/pylangacq) (The original repo can be found here: [pylangacq](https://github.com/jacksonllee/pylangacq)) using pip:
```
git clone git@github.com:mitjanikolaus/pylangacq.git
cd pylangacq
source activate cf
pip install .
```

### Running neural networks

To train and evaluate the neural network models, some additional dependencies are required. You can install the
`cf_nn` enviroment using the [environment_nn.yml](environments/environment_nn.yml) file:

```
conda env create --file environments/environment_nn.yml
```

## Data

### CHILDES corpora
All English CHILDES corpora need to be downloaded from the
[CHILDES database](https://childes.talkbank.org/) and extracted to `~/data/CHILDES/`.

## Preprocessing

The CHILDES corpus data is loaded using [my fork of the pylangacq repo](https://github.com/mitjanikolaus/pylangacq)

To preprocess the data, once you've installed the [pylangacq](https://github.com/mitjanikolaus/pylangacq) library as
mentioned above, you can run:
```
python preprocess.py
```
This preprocessed all corpora that are conversational (have child AND caregiver transcripts), and are in English.

Afterwards, annotate speech-relatedness and intelligibility:
```
python annotate.py
```

Finally, the utterances need to be annotated with speech acts. Use the method `crf_annotate` from the following
repo: [childes-speech-acts](https://github.com/mitjanikolaus/childes-speech-acts).
```
python crf_annotate.py --model checkpoint_full_train --data ~/data/communicative_feedback/utterances_annotated.csv --out ~/data/communicative_feedback/utterances_with_speech_acts.csv --use-pos --use-bi-grams --use-repetitions
```

## Analyses

First, extract the micro conversations for both analyses (speech-likeness and intelligibility):
```
python extract_micro_conversations.py --discard-non-speech-related-utterances
```

The script `analysis_grammaticality.py` performs the analyses and produce the results plots.

The script `analysis_grammaticality_glm.py` is used for the GLM analyses of the results.


## Annotation Scheme

### Intelligibility

The [CHAT coding scheme](https://talkbank.org/manuals/CHAT.pdf) offers multple ways to transcribe unintelligible
utterances. Here we list all cases that were most commonly used (the different corpora considered in our study make very
varying use of the different codes).

| Code       | Description                                                                                                                                |
|------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| xxx        | Unintelligible Speech                                                                                                                      |
| yyy        | Phonological Coding                                                                                                                        |
| @p         | Phonological Fragments (see above), e.g., bababa@p                                                                                         |
| \&         | Phonological Fragments (see above), e.g., \&baba                                                                                           |
| @b         | Babbling, e.g., bababa@b                                                                                                                   |
| \&=event   | Format used to transcribe ``simple events'', sometimes used for unintelligible speech: \&=babbling, \&=vocalizes                           |
| [=! event] | Format used to transcribe ``paralinguistic events'', sometimes used for unintelligible speech: [=! babbling], [=! voc]                     |
| word | Sometimes babbling is not explicitely, marked, but just transcribed as a normal word. We catch the most common cases e.g., "baba", "bababa" |





## Acknowledgements
Thanks to the authors of the pylangacq repo: 

Lee, Jackson L., Ross Burkholder, Gallagher B. Flinn, and Emily R. Coppess. 2016.
[Working with CHAT transcripts in Python](https://jacksonllee.com/papers/lee-etal-2016-pylangacq.pdf).
Technical report [TR-2016-02](https://newtraell.cs.uchicago.edu/research/publications/techreports/TR-2016-02),
Department of Computer Science, University of Chicago.
