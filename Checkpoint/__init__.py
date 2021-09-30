import pathlib

_CHECKPOINT_FOLDER = pathlib.Path(__file__).parent.absolute()
print("checkpoint folder:", _CHECKPOINT_FOLDER)


def get_checkpoint_folder():
    global _CHECKPOINT_FOLDER
    return _CHECKPOINT_FOLDER


def set_checkpoint_folder(path):
    global _CHECKPOINT_FOLDER
    _CHECKPOINT_FOLDER = path

    print("overriding checkpoint folder:", _CHECKPOINT_FOLDER)
