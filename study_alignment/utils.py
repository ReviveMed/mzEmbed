import os

def get_dropbox_dir():
    my_dir = '/Users/jonaheaton/Library/CloudStorage/Dropbox-ReviveMed/Jonah Eaton'
    if not os.path.exists(my_dir):
        my_dir = '/Users/jonaheaton/ReviveMed Dropbox/Jonah Eaton'
    if not os.path.exists(my_dir):
        raise ValueError('Dropbox directory not found')

    return my_dir