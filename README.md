# Automatic Detection of Creativity in Translation

This repository contains the code for my masters thesis from the Erasmus Mundus Masters in Language and Communication Technologies, developed at the University of Groningen and Charles University.

The data is not publicly released; this repository only includes dummy data for the story '2BR02B'.

In a nutshell, the thesis explores the automatic detection of creative potential in translation using various techniques:
- Cosine distance between source text and translation.
- Diversity between different translated versions of the source text into the same language.
- LLM prompting.
- Text classifier trained on synthetic data constructed using the best of the above methods.


## Setup
All of the steps assume that you are using a virtual environment as described below

### Virtual environment
Making sure that your python version is =<3.10.4,

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To deactivate the virtual environment, use the following command:
deactivate

### Installing external tools
If not already included as git submodules in the external_tools folder, add external git repositories as such:
```
mkdir external_tools
cd external_tools
git add submodule https://github.com/thompsonb/vecalign.git
git add submodule https://github.com/facebookresearch/LASER.git
git add submodule https://github.com/google-research/bleurt.git
```

#### LASER embeddings

1. Download the LASER models.
For this, move into the desired path and then run:
```
cd [REPO_PATH]
./external_tools/LASER/nllb/download_models.sh fake_lang
```
Note: fake_lang is a dummy argument to stop the script from downloading all the language-specific models that are available.

2. Manually introduce the path where the models are downloaded into external_tools/LASER/tasks/embed/embed.sh

3. set LASER environment variable
```
export LASER=[REPO_PATH]/external_tools/LASER
```

4. Install LASER external tools
Now we need to move into the LASER directory
```
cd external_tools/LASER
./install_external_tools.sh
```

#### BLEURT
The model needs to be downloaded manually.

```
cd external_tools
wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip .
unzip BLEURT-20.zip
```


### Login tokens
There are three main login tokens used in this repo to access different APIs: Anthropic, HuggingFace, and OpenAI.

To use the provided bash scripts, you should store them as follows:

```
root
...
|_ login_tokens
    |_ anthropic.txt
    |_ huggingface.txt
    |_ openai.txt
...
```

Where each of the .txt files contains only the corresponding API key.



## Using vecalign 
The script align_translations.sh takes care of all the process of aligning and merging these alignments for the books used in this investigation.

To add/remove books, modify the 'books' list within the script, and in the case of adding a book, also add the raw text files in the data/books folder.



## Recreating experiments
All experiments have a corresponding bash script that should take care of every step needed to reproduce an experiment.
These are found in the corresponding folder of the /src directory.

To reuse the bash scripts, a few modifications need to be done:
- Replace the `$VENVDIR` , `$DATADIR` and `$SRCDIR` paths with your own absolute paths.
- Provide the required data in its corresponding path.
- Include your own login tokens, as explained above.


## Note on data availability
The data included in this repository is only a dummy version including the versions of '2BR02B' by Kurt Vonnegut that are publicly available. This is intended to provide an example of the file structure and of the structure of the different parts of the data, and to make the code runnable. For the sake of showing the file structure, some empty folders remain where only content with other stories is stored.

Experimentation results are complete except for the creativity estimations of the translator_diversity folder, where some columns including copyrighted work have been removed. This leaves the final classifications and their corresponding evaluations untouched.