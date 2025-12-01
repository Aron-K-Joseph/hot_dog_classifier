<<<<<<< HEAD
# hot_dog_classifier
=======
# Hotdog vs. Not Hotdog Classifier

A PyTorch image classifier that distinguishes hotdog from not hotdog.

## Setup

Create and activate a virtual environment:

    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

## Dataset Structure

The script expects your dataset inside a folder named `data/` at the project root.

It must follow this structure:

    data/
        train/
            hotdog/
                img001.jpg
                img002.png
                ...
            nothotdog/
                img101.jpg
                img102.jpg
                ...
        test/
            hotdog/
                img001.jpg
                img002.png
                ...
            nothotdog/
                img101.jpg
                img102.jpg
                ...

Each subfolder becomes a class label. 

## Training the Model

To start training:

    python train.py

A trained model file (hotdog_model.pth) will be saved automatically.

## Using the website locally

To utilize the web interface, first run the app.py folder, which will begin running the Python backend

    python app.py

Once completed, open the index.html file with any desired local server hosting method.
>>>>>>> d8ad69f82d980d62ca1521e2810176a8ae7e6828
