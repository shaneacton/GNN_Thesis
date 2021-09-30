import pathlib

CHECKPOINT_FOLDER = pathlib.Path(__file__).parent.absolute()
print("checkpoint folder:", CHECKPOINT_FOLDER)


def set_checkpoint_folder(path):
    global CHECKPOINT_FOLDER
    CHECKPOINT_FOLDER = path

    print("overriding checkpoint folder:", CHECKPOINT_FOLDER)
