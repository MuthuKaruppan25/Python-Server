import pandas as pd
import numpy as np
from sklearn import __version__ as sklearn_version
import joblib
from flask import __version__ as flask_version
from flask_cors import __version__ as flask_cors_version
from pymongo import __version__ as pymongo_version

print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("scikit-learn:", sklearn_version)
print("joblib:", joblib.__version__)
print("flask:", flask_version)
print("flask-cors:", flask_cors_version)
print("pymongo:", pymongo_version)