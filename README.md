# communicative feedback


## Data

### CHILDES corpora
All CHILDES corpora listed in 'preprocess.py' need to be downloaded from the
[CHILDES database](https://childes.talkbank.org/) and extracted to `~/data/CHILDES/`.

### Turn taking meta-analysis
The [data](data/MA%20turn-taking.csv) of the [systematic review on turn-taking](https://psyarxiv.com/3bak6) was 
downloaded from the corresponding
[OSF project](https://osf.io/wkceb/?view_only=9cca387b49ef427fa9740cb94c3fbd5c).

## Preprocessing

The CHILDES corpus data is loaded using [my fork of the pylangacq repo](https://github.com/mitjanikolaus/pylangacq)
(The original repo can be found here: [pylangacq](https://github.com/jacksonllee/pylangacq))

To preprocess the data, install the [pylangacq](https://github.com/mitjanikolaus/pylangacq) library
and run:
```
preprocess.py
```

Afterwards the utterances need to be annotated with speech acts. Using the method `crf_annotate` from the following
repo: [Automatic speech act annotation](https://github.com/mitjanikolaus/childes-speech-acts/tree/new-data-loading).
```
crf_annotate --model checkpoints/crf_full_train --data ~/data/communicative_feedback/utterances.p --out ~/data/communicative_feedback/utterances_with_speech_acts.p --use-pos --use-bi-grams --use-repetitions
```

Finally, annotate speech-relatedness and intelligibility:
```
annotate.py --rule-based-intelligibility
```

## Analyses

The scripts `analysis_reproduce_warlaumont.py` and `analysis_intelligibility` perform the respective analyses and
produce the plots.

The notebook `analysis_intelligibility_glm.ipynb` is used for the GLM analyses, which are performed with R.

## Acknowledgements
Thanks to the authors of the pylangacq repo: 

Lee, Jackson L., Ross Burkholder, Gallagher B. Flinn, and Emily R. Coppess. 2016.
[Working with CHAT transcripts in Python](https://jacksonllee.com/papers/lee-etal-2016-pylangacq.pdf).
Technical report [TR-2016-02](https://newtraell.cs.uchicago.edu/research/publications/techreports/TR-2016-02),
Department of Computer Science, University of Chicago.
