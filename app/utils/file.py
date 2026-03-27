import os
from app.config import Config


def allowed_file(filename):
    return (
        "." in filename
        and os.path.splitext(filename)[1][1:] in Config.ALLOWED_EXTENSIONS
    )
