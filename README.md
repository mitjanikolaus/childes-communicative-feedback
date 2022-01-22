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
and run `preprocess.py`.

Afterwards the utterances need to be annotated with speech acts: [Automatic speech act annotation](https://github.com/mitjanikolaus/childes-speech-acts/tree/new-data-loading).

Finally, run `annotate.py` to annotate speech-relatedness and intelligibility.

## Acknowledgements
Thanks to the authors of the pylangacq repo: 

Lee, Jackson L., Ross Burkholder, Gallagher B. Flinn, and Emily R. Coppess. 2016.
[Working with CHAT transcripts in Python](https://jacksonllee.com/papers/lee-etal-2016-pylangacq.pdf).
Technical report [TR-2016-02](https://newtraell.cs.uchicago.edu/research/publications/techreports/TR-2016-02),
Department of Computer Science, University of Chicago.
