import os
from dotenv import load_dotenv

load_dotenv()


ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
