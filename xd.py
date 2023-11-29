from flask import Flask, render_template, session, request, redirect, url_for, send_file
from flask_session import Session
import numpy as np
import os
from copy import copy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from werkzeug.utils import secure_filename
from sklearn.model_selection import StratifiedKFold

