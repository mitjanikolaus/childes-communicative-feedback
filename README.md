# CHILDES communicative feedback

This repository contains code accompanying the following paper:

**Communicative Feedback as a Mechanism Supporting the Production of Intelligible Speech in Early Childhood** \
*In Proceedings of the 44th Annual Meeting of the Cognitive Science Society. (2022)* \
Mitja Nikolaus, Laurent Prévot and Abdellah Fourtassi

## Data

### CHILDES corpora
All CHILDES corpora listed in 'preprocess.py' need to be downloaded from the
[CHILDES database](https://childes.talkbank.org/) and extracted to `~/data/CHILDES/`.

The following corpora were considered in the analyses:
Bernstein, Bloom, Braunwald, Brent, Edinburgh, Gleason, MPI-EVA-Manchester, MacWhinney, McCune, McMillan, Nelson, NewmanRatner, Peters, Providence, Rollins, Sachs, Snow, Soderstrom, Thomas, Tommerdahl, VanHouten, Weist

## Preprocessing

The CHILDES corpus data is loaded using [my fork of the pylangacq repo](https://github.com/mitjanikolaus/pylangacq)
(The original repo can be found here: [pylangacq](https://github.com/jacksonllee/pylangacq))

To preprocess the data, install the [pylangacq](https://github.com/mitjanikolaus/pylangacq) library
and run:
```
preprocess.py
```

Afterwards the utterances need to be annotated with speech acts. Use the method `crf_annotate` from the following
branch: [automatic speech act annotation](https://github.com/mitjanikolaus/childes-speech-acts/tree/new-data-loading).
```
crf_annotate --model checkpoints/crf_full_train --data ~/data/communicative_feedback/utterances.p --out ~/data/communicative_feedback/utterances_with_speech_acts.p --use-pos --use-bi-grams --use-repetitions
```

Finally, annotate speech-relatedness and intelligibility:
```
annotate.py
```

## Analyses

The scripts `analysis_reproduce_warlaumont.py` and `analysis_intelligibility` perform the respective analyses and
produce the plots.

The scripts `analysis_reproduce_warlaumont_glm.py` and `analysis_intelligibility_glm.py` are used for the GLM analyses
of the results.


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
