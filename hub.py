from config import Config

def allow_file(filename):
    return "." in filename and filename.rsplit('.', 1)[1] in Config.ALLOWED_EXTENSIONS