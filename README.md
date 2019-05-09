# rovigappy

NB is required a version of python >= 3.5

Using dataset from https://datahub.io/sports-data/  by following this guide:

    https://datahub.io/sports-data/italian-serie-a#python

The library datapackage is required to retrieve the dataset
so you can run this line before executing create_dataset.py:

    pip install datapackage
    
Is recommended though to setup the whole system by running:

    pip install -r requirements.txt

To develop on this project it is preferable to install Git Flow and open
your own feature starting from develop.
To do so run:

    sudo apt-get install git-flow

Navigate from terminal to the repository and run:

    git flow init
    
Leave default configuration by pressing enter when prompted.
Checkout branch develop and run this command to open up a new feature:

    git flow feature start <MY_FEATURE_NAME>
    
Each feature should be named with the task is being developed in it.
When task is exhausted is fine to close the feature on develop to test
functionality and then merge it into master.

For more information about Git Flow check official documentation here:
https://danielkummer.github.io/git-flow-cheatsheet/
