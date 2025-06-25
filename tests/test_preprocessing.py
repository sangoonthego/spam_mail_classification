import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import clean_text

sample = "Congratulations! You've won $1000. Click here to claim: https://spam.link"
print(clean_text(sample))