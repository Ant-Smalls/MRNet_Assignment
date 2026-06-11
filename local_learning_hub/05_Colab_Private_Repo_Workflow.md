# Colab & Private Repo Workflow

This document explains how to use Google Colab to train the models using your private GitHub repository, completely protecting your work from the shared group project until you are ready to share it.

## The Dual-Remote Setup
Your local folder (`/Users/jinishrajan/ucd/Medimaging/`) is connected to TWO remote GitHub repositories:
1. **origin**: The shared team repository.
2. **mine**: Your private repository (`Dublindeveloper/mrnet-learning`).

Currently, all the brilliant code you have written (`train.py`, `explainability.py`, etc.) has only been pushed to **mine**. The team cannot see it.

## How the Colab Notebook Works Safely
When you open `MRNet_Colab_Training.ipynb` in Google Colab, here is what is happening:
1. The notebook runs a `git clone` command using your Personal Access Token (PAT).
2. It clones the code **exclusively from your private repo** (`mine`). 
3. It mounts your Google Drive.
4. As the models train, the heavy `.pth` checkpoint files are saved directly into your Google Drive, **not** into GitHub.

Because Colab is pulling from your private repo, and saving files to your private Drive, there is **zero risk** of accidentally messing up the team's shared repository.

## How to Eventually Share with the Group
When you are ready to share the working pipeline with your computer engineer teammates, follow these steps to avoid a "big issue" or messy merge conflicts:

1. **Do NOT try to merge your private repo into the shared repo automatically.** 
2. Instead, simply give your team the URL to your private GitHub repository (you will need to invite them as collaborators or make it public for a day).
3. Say: *"Hey guys, I built a prototype training pipeline here. Feel free to copy these python files over to our official group repo!"*
4. Your teammates can then manually copy the polished `train.py`, `evaluation.py`, and `data_preprocessing.py` files into their own machines and push them to the official group repo themselves.

This keeps you perfectly isolated, protects your learning space, and gives the team clean, working code when you decide to hand it over!
