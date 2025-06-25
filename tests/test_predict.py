import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import predict_email

email = "Congratulations! You've won a free ticket. Click here!"
result = predict_email(email)
print("Result:", result)