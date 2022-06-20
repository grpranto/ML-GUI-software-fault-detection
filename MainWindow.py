from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QLabel, QGroupBox, \
    QHBoxLayout, QCheckBox, QDialog, QGridLayout, QRadioButton, QWidget, QProgressBar, QSplashScreen, QTableWidget,QTableWidgetItem
import sys
from PyQt5.QtCore import QRect, QFile, Qt
import pandas as pd
from pandas import DataFrame
from PyQt5 import QtCore
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
#from keras.models import Sequential #building NN layer by layer
import time
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

dataset_name = ""
selected = ""
accuracy = 0.0
conf = ""
data = ""

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.title = "Effect of feature selection in Software fault detection"
        self.setFixedSize(510, 480)
        self.InitWindow()


    def button(self):
        calButton = QPushButton("Calculate", self)
        calButton.setGeometry(QRect(375, 10, 111, 28))
        calButton.clicked.connect(self.on_click_calculate)
        #calButton.clicked.connect(self.calculate)

        datasetButton = QPushButton("Input Dataset", self)
        datasetButton.setGeometry(QRect(250, 10, 111, 28))
        datasetButton.clicked.connect(self.on_click_inputDataset)

        calButton.setStyleSheet("QPushButton { background-color: #027d0d; color: white; }"
                          "QPushButton:pressed { background-color: #000000; color:white }")

        datasetButton.setStyleSheet("QPushButton { background-color: #0e568a; color: white; }"
                                "QPushButton:pressed { background-color: #000000; color:white }")


    def on_click_inputDataset(self):
        self.openFileNameDialog()
        if (dataset_name != ""):
            self.SW = ThirdWindow()
        '''if(dataset_name != ""):
            df = pd.read_excel(dataset_name)
            yes = (df['Defective']=='Y').sum()
            no = (df['Defective']=='N').sum()
            labels = 'Defective', 'Non-Defective'
            sizes = [yes, no]
            colors = ['gold','lightskyblue']
            explode = (0.1, 0)  # explode 1st slice

            # Plot
            plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                    autopct='%1.1f%%', shadow=True, startangle=140)
            plt.title("Dataset: " + dataset_name +"\nColumns: " + str(len(df.columns)) + "   Rows: " + str(yes+no)+ "\nDefective: " + str(yes) + "   Non-Defective: " + str(no))
            plt.axis('equal')
            plt.show()'''


    def on_click_calculate(self):
        flag = True
        if(selected != "" and dataset_name != ""):
            if (dataset_name.find('CM1')!=-1):
                print(selected)
                #DECISION TREE-----------------------------------
                if (selected == "check13"):
                    dataset = "CM1_20_IG.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.DT(X, y)
                elif (selected == "check14"):
                    dataset = "CM1_20_REL.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.DT(X, y)
                elif (selected == "check18"):
                    dataset = "CM1_20_CHI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.DT(X, y)
                elif (selected == "check19"):
                    dataset = "CM1_20_CHI2.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.DT(X, y)
                elif (selected == "check110"):
                    dataset = "CM1_20_FI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.DT(X, y)


                #Random Forest------------------------

                elif (selected == "check23"):
                    dataset = "CM1_20_IG.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.RF(X, y)
                elif (selected == "check24"):
                    dataset = "CM1_20_REL.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.RF(X, y)
                elif (selected == "check28"):
                    dataset = "CM1_20_CHI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.RF(X, y)
                elif (selected == "check29"):
                    dataset = "CM1_20_CHI2.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.RF(X, y)
                elif (selected == "check210"):
                    dataset = "CM1_20_FI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.RF(X, y)

                # Naive Bayes------------------------

                elif (selected == "check53"):
                    dataset = "CM1_20_IG.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.NB(X, y)
                elif (selected == "check54"):
                    dataset = "CM1_20_REL.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.NB(X, y)
                elif (selected == "check58"):
                    dataset = "CM1_20_CHI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.NB(X, y)
                elif (selected == "check59"):
                    dataset = "CM1_20_CHI2.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.NB(X, y)
                elif (selected == "check510"):
                    dataset = "CM1_20_FI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.NB(X, y)

                # Logistic Regression------------------------

                elif (selected == "check63"):
                    dataset = "CM1_20_IG.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.LR(X, y)
                elif (selected == "check64"):
                    dataset = "CM1_20_REL.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.LR(X, y)
                elif (selected == "check68"):
                    dataset = "CM1_20_CHI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.LR(X, y)
                elif (selected == "check69"):
                    dataset = "CM1_20_CHI2.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.LR(X, y)
                elif (selected == "check610"):
                    dataset = "CM1_20_FI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.LR(X, y)


                elif(dataset_name == "CM1.xls"):
                    df = pd.read_excel(dataset_name)

                    if(selected == "check1"):
                        end = 36
                        X = df.values[:, 0:end]
                        y = df.Defective
                        self.DT(X, y)

                    if (selected == "check2"):
                        end = 36
                        X = df.values[:, 0:end]
                        y = df.Defective
                        self.RF(X, y)

                    if (selected == "check5"):
                        end = 36
                        X = df.values[:, 0:end]
                        y = df.Defective
                        self.NB(X, y)

                    if (selected == "check6"):
                        end = 36
                        X = df.values[:, 0:end]
                        y = df.Defective
                        self.LR(X, y)




            elif (dataset_name.find('MW1') != -1):
                print(selected)
                # DECISION TREE-----------------------------------
                if (selected == "check13"):
                    dataset = "MW1_20_IG.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.DT(X, y)
                elif (selected == "check14"):
                    dataset = "MW1_20_REL.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.DT(X, y)
                elif (selected == "check18"):
                    dataset = "MW1_20_CHI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.DT(X, y)
                elif (selected == "check19"):
                    dataset = "MW1_20_CHI2.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.DT(X, y)
                elif (selected == "check110"):
                    dataset = "MW1_20_FI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.DT(X, y)


                # Random Forest------------------------

                elif (selected == "check23"):
                    dataset = "MW1_20_IG.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.RF(X, y)
                elif (selected == "check24"):
                    dataset = "MW1_20_REL.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.RF(X, y)
                elif (selected == "check28"):
                    dataset = "MW1_20_CHI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.RF(X, y)
                elif (selected == "check29"):
                    dataset = "MW1_20_CHI2.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.RF(X, y)
                elif (selected == "check210"):
                    dataset = "MW1_20_FI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.RF(X, y)

                # Naive Bayes------------------------

                elif (selected == "check53"):
                    dataset = "MW1_20_IG.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.NB(X, y)
                elif (selected == "check54"):
                    dataset = "MW1_20_REL.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.NB(X, y)
                elif (selected == "check58"):
                    dataset = "MW1_20_CHI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.NB(X, y)
                elif (selected == "check59"):
                    dataset = "MW1_20_CHI2.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.NB(X, y)
                elif (selected == "check510"):
                    dataset = "MW1_20_FI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.NB(X, y)

                # Logistic Regression------------------------

                elif (selected == "check63"):
                    dataset = "MW1_20_IG.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.LR(X, y)
                elif (selected == "check64"):
                    dataset = "MW1_20_REL.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.LR(X, y)
                elif (selected == "check68"):
                    dataset = "MW1_20_CHI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.LR(X, y)
                elif (selected == "check69"):
                    dataset = "MW1_20_CHI2.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.LR(X, y)
                elif (selected == "check610"):
                    dataset = "MW1_20_FI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.LR(X, y)


                elif (dataset_name == "MW1.xls"):
                    df = pd.read_excel(dataset_name)

                    if (selected == "check1"):
                        end = 36
                        X = df.values[:, 0:end]
                        y = df.Defective
                        self.DT(X, y)

                    if (selected == "check2"):
                        end = 36
                        X = df.values[:, 0:end]
                        y = df.Defective
                        self.RF(X, y)

                    if (selected == "check5"):
                        end = 36
                        X = df.values[:, 0:end]
                        y = df.Defective
                        self.NB(X, y)

                    if (selected == "check6"):
                        end = 36
                        X = df.values[:, 0:end]
                        y = df.Defective
                        self.LR(X, y)

            elif (dataset_name.find('PC2') != -1):
                print(selected)
                # DECISION TREE-----------------------------------
                if (selected == "check13"):
                    dataset = "PC2_20_IG.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.DT(X, y)
                elif (selected == "check14"):
                    dataset = "PC2_20_REL.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.DT(X, y)
                elif (selected == "check18"):
                    dataset = "PC2_20_CHI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.DT(X, y)
                elif (selected == "check19"):
                    dataset = "PC2_20_CHI2.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.DT(X, y)
                elif (selected == "check110"):
                    dataset = "PC2_20_FI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.DT(X, y)


                # Random Forest------------------------

                elif (selected == "check23"):
                    dataset = "PC2_20_IG.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.RF(X, y)
                elif (selected == "check24"):
                    dataset = "PC2_20_REL.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.RF(X, y)
                elif (selected == "check28"):
                    dataset = "PC2_20_CHI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.RF(X, y)
                elif (selected == "check29"):
                    dataset = "PC2_20_CHI2.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.RF(X, y)
                elif (selected == "check210"):
                    dataset = "PC2_20_FI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.RF(X, y)

                # Naive Bayes------------------------

                elif (selected == "check53"):
                    dataset = "PC2_20_IG.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.NB(X, y)
                elif (selected == "check54"):
                    dataset = "PC2_20_REL.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.NB(X, y)
                elif (selected == "check58"):
                    dataset = "PC2_20_CHI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.NB(X, y)
                elif (selected == "check59"):
                    dataset = "PC2_20_CHI2.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.NB(X, y)
                elif (selected == "check510"):
                    dataset = "PC2_20_FI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.NB(X, y)

                # Logistic Regression------------------------

                elif (selected == "check63"):
                    dataset = "PC2_20_IG.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.LR(X, y)
                elif (selected == "check64"):
                    dataset = "PC2_20_REL.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.LR(X, y)
                elif (selected == "check68"):
                    dataset = "PC2_20_CHI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.LR(X, y)
                elif (selected == "check69"):
                    dataset = "PC2_20_CHI2.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.LR(X, y)
                elif (selected == "check610"):
                    dataset = "PC2_20_FI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.LR(X, y)


                elif (dataset_name == "PC2.xls"):
                    df = pd.read_excel(dataset_name)

                    if (selected == "check1"):
                        end = 36
                        X = df.values[:, 0:end]
                        y = df.Defective
                        self.DT(X, y)

                    if (selected == "check2"):
                        end = 36
                        X = df.values[:, 0:end]
                        y = df.Defective
                        self.RF(X, y)

                    if (selected == "check5"):
                        end = 36
                        X = df.values[:, 0:end]
                        y = df.Defective
                        self.NB(X, y)

                    if (selected == "check6"):
                        end = 36
                        X = df.values[:, 0:end]
                        y = df.Defective
                        self.LR(X, y)



            elif (dataset_name.find('PC4') != -1):
                print(selected)
                # DECISION TREE-----------------------------------
                if (selected == "check13"):
                    dataset = "PC4_20_IG.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.DT(X, y)
                elif (selected == "check14"):
                    dataset = "PC4_20_REL.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.DT(X, y)
                elif (selected == "check18"):
                    dataset = "PC4_20_CHI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.DT(X, y)
                elif (selected == "check19"):
                    dataset = "PC4_20_CHI2.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.DT(X, y)
                elif (selected == "check110"):
                    dataset = "PC4_20_FI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.DT(X, y)


                # Random Forest------------------------

                elif (selected == "check23"):
                    dataset = "PC4_20_IG.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.RF(X, y)
                elif (selected == "check24"):
                    dataset = "PC4_20_REL.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.RF(X, y)
                elif (selected == "check28"):
                    dataset = "PC4_20_CHI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.RF(X, y)
                elif (selected == "check29"):
                    dataset = "PC4_20_CHI2.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.RF(X, y)
                elif (selected == "check210"):
                    dataset = "PC4_20_FI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.RF(X, y)

                # Naive Bayes------------------------

                elif (selected == "check53"):
                    dataset = "PC4_20_IG.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.NB(X, y)
                elif (selected == "check54"):
                    dataset = "PC4_20_REL.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.NB(X, y)
                elif (selected == "check58"):
                    dataset = "PC4_20_CHI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.NB(X, y)
                elif (selected == "check59"):
                    dataset = "PC4_20_CHI2.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.NB(X, y)
                elif (selected == "check510"):
                    dataset = "PC4_20_FI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.NB(X, y)

                # Logistic Regression------------------------

                elif (selected == "check63"):
                    dataset = "PC4_20_IG.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.LR(X, y)
                elif (selected == "check64"):
                    dataset = "PC4_20_REL.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.LR(X, y)
                elif (selected == "check68"):
                    dataset = "PC4_20_CHI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.LR(X, y)
                elif (selected == "check69"):
                    dataset = "PC4_20_CHI2.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.LR(X, y)
                elif (selected == "check610"):
                    dataset = "PC4_20_FI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.LR(X, y)


                elif (dataset_name == "PC4.xls"):
                    df = pd.read_excel(dataset_name)

                    if (selected == "check1"):
                        end = 36
                        X = df.values[:, 0:end]
                        y = df.Defective
                        self.DT(X, y)

                    if (selected == "check2"):
                        end = 36
                        X = df.values[:, 0:end]
                        y = df.Defective
                        self.RF(X, y)

                    if (selected == "check5"):
                        end = 36
                        X = df.values[:, 0:end]
                        y = df.Defective
                        self.NB(X, y)

                    if (selected == "check6"):
                        end = 36
                        X = df.values[:, 0:end]
                        y = df.Defective
                        self.LR(X, y)

            elif (dataset_name.find('KC3') != -1):
                print(selected)
                # DECISION TREE-----------------------------------
                if (selected == "check13"):
                    dataset = "KC3_20_IG.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.DT(X, y)
                elif (selected == "check14"):
                    dataset = "KC3_20_REL.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.DT(X, y)
                elif (selected == "check18"):
                    dataset = "KC3_20_CHI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.DT(X, y)
                elif (selected == "check19"):
                    dataset = "KC3_20_CHI2.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.DT(X, y)
                elif (selected == "check110"):
                    dataset = "KC3_20_FI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.DT(X, y)


                # Random Forest------------------------

                elif (selected == "check23"):
                    dataset = "KC3_20_IG.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.RF(X, y)
                elif (selected == "check24"):
                    dataset = "KC3_20_REL.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.RF(X, y)
                elif (selected == "check28"):
                    dataset = "KC3_20_CHI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.RF(X, y)
                elif (selected == "check29"):
                    dataset = "KC3_20_CHI2.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.RF(X, y)
                elif (selected == "check210"):
                    dataset = "KC3_20_FI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.RF(X, y)

                # Naive Bayes------------------------

                elif (selected == "check53"):
                    dataset = "KC3_20_IG.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.NB(X, y)
                elif (selected == "check54"):
                    dataset = "KC3_20_REL.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.NB(X, y)
                elif (selected == "check58"):
                    dataset = "KC3_20_CHI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.NB(X, y)
                elif (selected == "check59"):
                    dataset = "KC3_20_CHI2.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.NB(X, y)
                elif (selected == "check510"):
                    dataset = "KC3_20_FI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.NB(X, y)

                # Logistic Regression------------------------

                elif (selected == "check63"):
                    dataset = "KC3_20_IG.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.LR(X, y)
                elif (selected == "check64"):
                    dataset = "KC3_20_REL.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.LR(X, y)
                elif (selected == "check68"):
                    dataset = "KC3_20_CHI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.LR(X, y)
                elif (selected == "check69"):
                    dataset = "KC3_20_CHI2.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.LR(X, y)
                elif (selected == "check610"):
                    dataset = "KC3_20_FI.csv"
                    df = pd.read_csv(dataset)
                    end = 18
                    X = df.values[:, 0:end]
                    y = df.Defective
                    self.LR(X, y)


                elif (dataset_name == "KC3.xls"):
                    df = pd.read_excel(dataset_name)

                    if (selected == "check1"):
                        end = 36
                        X = df.values[:, 0:end]
                        y = df.Defective
                        self.DT(X, y)

                    if (selected == "check2"):
                        end = 36
                        X = df.values[:, 0:end]
                        y = df.Defective
                        self.RF(X, y)

                    if (selected == "check5"):
                        end = 36
                        X = df.values[:, 0:end]
                        y = df.Defective
                        self.NB(X, y)

                    if (selected == "check6"):
                        end = 36
                        X = df.values[:, 0:end]
                        y = df.Defective
                        self.LR(X, y)


        else:
            self.label.setStyleSheet('color: red')
            self.label.setText(str("**Dataset, classification and/or technique must be selected first"))
            flag = False

        if flag == True:
            self.SW = SecondWindow()


    def openFileNameDialog(self):
        global dataset_name
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Dataset", "",
                                                "XLS Files (*.xls);;CSV Files (*.csv)", options=options)
        dataset_name = fileName
        data = dataset_name
        dataset_name = fileName.split('/')[-1]

    #FIRST ROW
    def CreateLayout(self):
        self.setContentsMargins(0, 30, 0, 0)
        self.groupBox = QGroupBox("Select a classifier (required)")
        self.groupBox.setFont(QtGui.QFont("Sanserif", 13))
        vboxLayout = QVBoxLayout()

        self.check1 = QCheckBox("Decision Tree")
        self.check1.setIcon(QtGui.QIcon("DT.png"))
        self.check1.setIconSize(QtCore.QSize(20, 20))
        self.check1.setFont(QtGui.QFont("Sanserif", 10))
        self.check1.toggled.connect(self.onCheckbox_toggled)
        vboxLayout.addWidget(self.check1)

        self.check2 = QCheckBox("Random Forest")
        self.check2.setIcon(QtGui.QIcon("RF.png"))
        self.check2.setIconSize(QtCore.QSize(20, 20))
        self.check2.setFont(QtGui.QFont("Sanserif", 10))
        self.check2.toggled.connect(self.onCheckbox_toggled)
        vboxLayout.addWidget(self.check2)

        self.check5 = QCheckBox("Naive Bayes")
        self.check5.setIcon(QtGui.QIcon("NB.png"))
        self.check5.setIconSize(QtCore.QSize(20, 20))
        self.check5.setFont(QtGui.QFont("Sanserif", 10))
        self.check5.toggled.connect(self.onCheckbox_toggled)
        vboxLayout.addWidget(self.check5)

        self.check6 = QCheckBox("Logistic Regression")
        self.check6.setIcon(QtGui.QIcon("LR.png"))
        self.check6.setIconSize(QtCore.QSize(20, 20))
        self.check6.setFont(QtGui.QFont("Sanserif", 10))
        self.check6.toggled.connect(self.onCheckbox_toggled)
        vboxLayout.addWidget(self.check6)

        self.check7 = QCheckBox("Neural Network")
        self.check7.setIcon(QtGui.QIcon("NN.png"))
        self.check7.setIconSize(QtCore.QSize(20, 20))
        self.check7.setFont(QtGui.QFont("Sanserif", 10))
        self.check7.toggled.connect(self.onCheckbox_toggled)
        vboxLayout.addWidget(self.check7)
        #-------------------------------------------------------

        self.groupBox1 = QGroupBox("Select a technique (optional)")
        self.groupBox1.setFont(QtGui.QFont("Sanserif", 13))
        vboxLayout1 = QVBoxLayout()

        self.check3 = QCheckBox("Information Gain")
        self.check3.setIcon(QtGui.QIcon("IG.png"))
        self.check3.setIconSize(QtCore.QSize(20, 20))
        self.check3.setFont(QtGui.QFont("Sanserif", 10))
        self.check3.toggled.connect(self.onCheckbox_toggled)
        vboxLayout1.addWidget(self.check3)

        self.check4 = QCheckBox("Relief")
        self.check4.setIcon(QtGui.QIcon("REL.png"))
        self.check4.setIconSize(QtCore.QSize(20, 20))
        self.check4.setFont(QtGui.QFont("Sanserif", 10))
        self.check4.toggled.connect(self.onCheckbox_toggled)
        vboxLayout1.addWidget(self.check4)

        self.check8 = QCheckBox("Chi Square")
        self.check8.setIcon(QtGui.QIcon("CHI2.png"))
        self.check8.setIconSize(QtCore.QSize(20, 20))
        self.check8.setFont(QtGui.QFont("Sanserif", 10))
        self.check8.toggled.connect(self.onCheckbox_toggled)
        vboxLayout1.addWidget(self.check8)

        self.check9 = QCheckBox("Chi Square with independence")
        self.check9.setIcon(QtGui.QIcon("CHI.png"))
        self.check9.setIconSize(QtCore.QSize(20, 20))
        self.check9.setFont(QtGui.QFont("Sanserif", 10))
        self.check9.toggled.connect(self.onCheckbox_toggled)
        vboxLayout1.addWidget(self.check9)

        self.check10 = QCheckBox("Feature Importance")
        self.check10.setIcon(QtGui.QIcon("FI.png"))
        self.check10.setIconSize(QtCore.QSize(20, 20))
        self.check10.setFont(QtGui.QFont("Sanserif", 10))
        self.check10.toggled.connect(self.onCheckbox_toggled)
        vboxLayout1.addWidget(self.check10)
        #------------------------------------------------------------

        self.groupBox.setLayout(vboxLayout)
        self.groupBox1.setLayout(vboxLayout1)



    def onCheckbox_toggled(self):
        global selected
        if self.check1.isChecked():
            if self.check3.isChecked():
                selected = "check13"
                self.label.setText(self.check1.text() + " and " + self.check3.text() + " are selected")
            elif self.check4.isChecked():
                selected = "check14"
                self.label.setText(self.check1.text() + " and " + self.check4.text() + " are selected")
            elif self.check8.isChecked():
                selected = "check18"
                self.label.setText(self.check1.text() + " and " + self.check8.text() + " are selected")
            elif self.check9.isChecked():
                selected = "check19"
                self.label.setText(self.check1.text() + " and " + self.check9.text() + " are selected")
            elif self.check10.isChecked():
                selected = "check110"
                self.label.setText(self.check1.text() + " and " + self.check10.text() + " are selected")
            else:
                selected = "check1"
                self.label.setText(self.check1.text() + " is selected")



        elif self.check2.isChecked():
            if self.check3.isChecked():
                selected = "check23"
                self.label.setText(self.check2.text() + " and " + self.check3.text() + " are selected")
            elif self.check4.isChecked():
                selected = "check24"
                self.label.setText(self.check2.text() + " and " + self.check4.text() + " are selected")
            elif self.check8.isChecked():
                selected = "check28"
                self.label.setText(self.check2.text() + " and " + self.check8.text() + " are selected")
            elif self.check9.isChecked():
                selected = "check29"
                self.label.setText(self.check2.text() + " and " + self.check9.text() + " are selected")
            elif self.check10.isChecked():
                selected = "check210"
                self.label.setText(self.check2.text() + " and " + self.check10.text() + " are selected")
            else:
                selected = "check2"
                self.label.setText(self.check2.text() + " is selected")

        elif self.check5.isChecked():
            if self.check3.isChecked():
                selected = "check53"
                self.label.setText(self.check5.text() + " and " + self.check3.text() + " are selected")
            elif self.check4.isChecked():
                selected = "check54"
                self.label.setText(self.check5.text() + " and " + self.check4.text() + " are selected")
            elif self.check8.isChecked():
                selected = "check58"
                self.label.setText(self.check5.text() + " and " + self.check8.text() + " are selected")
            elif self.check9.isChecked():
                selected = "check59"
                self.label.setText(self.check5.text() + " and " + self.check9.text() + " are selected")
            elif self.check10.isChecked():
                selected = "check510"
                self.label.setText(self.check5.text() + " and " + self.check10.text() + " are selected")
            else:
                selected = "check5"
                self.label.setText(self.check5.text() + " is selected")

        elif self.check6.isChecked():
            if self.check3.isChecked():
                selected = "check63"
                self.label.setText(self.check6.text() + " and " + self.check3.text() + " are selected")
            elif self.check4.isChecked():
                selected = "check64"
                self.label.setText(self.check6.text() + " and " + self.check4.text() + " are selected")
            elif self.check8.isChecked():
                selected = "check68"
                self.label.setText(self.check6.text() + " and " + self.check8.text() + " are selected")
            elif self.check9.isChecked():
                selected = "check69"
                self.label.setText(self.check6.text() + " and " + self.check9.text() + " are selected")
            elif self.check10.isChecked():
                selected = "check610"
                self.label.setText(self.check6.text() + " and " + self.check10.text() + " are selected")
            else:
                selected = "check6"
                self.label.setText(self.check6.text() + " is selected")

        elif self.check7.isChecked():
            if self.check3.isChecked():
                selected = "check73"
                self.label.setText(self.check7.text() + " and " + self.check3.text() + " are selected")
            elif self.check4.isChecked():
                selected = "check74"
                self.label.setText(self.check7.text() + " and " + self.check4.text() + " are selected")
            elif self.check8.isChecked():
                selected = "check78"
                self.label.setText(self.check7.text() + " and " + self.check8.text() + " are selected")
            elif self.check9.isChecked():
                selected = "check79"
                self.label.setText(self.check7.text() + " and " + self.check9.text() + " are selected")
            elif self.check10.isChecked():
                selected = "check710"
                self.label.setText(self.check7.text() + " and " + self.check10.text() + " are selected")
            else:
                selected = "check7"
                self.label.setText(self.check7.text() + " is selected")

        else:
            self.label.setText("")
            selected = ""



    def InitWindow(self):
        self.setWindowTitle(self.title)
        #self.setGeometry(self.left, self.top, self.width, self.height)
        self.button()
        self.setWindowIcon(QtGui.QIcon('icon.jpg'))
        self.CreateLayout()



        vbox = QVBoxLayout()

        vbox.addWidget(self.groupBox)
        vbox.addWidget(self.groupBox1)

        self.label = QLabel(self)
        self.label.setFont(QtGui.QFont("Sanserif", 10))
        vbox.addWidget(self.label)
        self.setLayout(vbox)
        self.show()


    def DT(self, X, y):
        global accuracy, conf, fpr, tpr, auc
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)  # 70% training and 30% test
        clf = DecisionTreeClassifier()
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
        accuracy = metrics.accuracy_score(y_test, y_pred)
        print("Classification Report\n", classification_report(y_test, y_pred))
        print("Confusion Matrix\n", confusion_matrix(y_test, y_pred))
        conf = confusion_matrix(y_test, y_pred)

        probs = clf.predict_proba(X_test)
        probs = probs[:, 1]
        auc = roc_auc_score(y_test, probs)
        print('AUC: %.2f' % auc)
        fpr, tpr, thresholds = roc_curve(y_test, probs, pos_label='Y')
        #self.ROC(fpr, tpr)

    def RF(self, X, y):
        global accuracy, conf, fpr, tpr, auc
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)  # 70% training and 30% test
        clf = RandomForestClassifier(n_estimators=100)
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
        print("Classification Report\n", classification_report(y_test, y_pred))
        print("Confusion Matrix\n", confusion_matrix(y_test, y_pred))
        conf = confusion_matrix(y_test, y_pred)

        probs = clf.predict_proba(X_test)
        probs = probs[:, 1]
        auc = roc_auc_score(y_test, probs)
        print('AUC: %.2f' % auc)
        fpr, tpr, thresholds = roc_curve(y_test, probs, pos_label='Y')
        #self.ROC(fpr, tpr)

    def NB(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=1)  # 70% training and 30% test
        clf = GaussianNB()
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        global accuracy, conf, fpr, tpr, auc
        accuracy = metrics.accuracy_score(y_test, y_pred)
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
        print("Classification Report\n", classification_report(y_test, y_pred))
        print("Confusion Matrix\n", confusion_matrix(y_test, y_pred))
        conf = confusion_matrix(y_test, y_pred)

        probs = clf.predict_proba(X_test)
        probs = probs[:, 1]
        auc = roc_auc_score(y_test, probs)
        print('AUC: %.2f' % auc)
        fpr, tpr, thresholds = roc_curve(y_test, probs, pos_label='Y')
        #self.ROC(fpr, tpr)

    def LR(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)  # 70% training and 30% test
        clf = LogisticRegression()
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        global accuracy, conf, fpr, tpr, auc
        accuracy = metrics.accuracy_score(y_test, y_pred)
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
        print("Classification Report\n", classification_report(y_test, y_pred))
        print("Confusion Matrix\n", confusion_matrix(y_test, y_pred))
        conf = confusion_matrix(y_test, y_pred)

        probs = clf.predict_proba(X_test)
        probs = probs[:, 1]
        auc = roc_auc_score(y_test, probs)
        print('AUC: %.2f' % auc)
        fpr, tpr, thresholds = roc_curve(y_test, probs, pos_label='Y')
        #self.ROC(fpr, tpr)







    def calculate(self):
        self.SW = SecondWindow()
        #self.SW.show()


class SecondWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Result")
        #self.setFixedSize(800, 600)
        self.setWindowIcon(QtGui.QIcon('icon1.png'))

        self.lab = QLabel("Accuracy = " + str(accuracy) + "\n\nConfusion Matrix" , self)
        #self.lab.move(50, 30)
        self.lab.setFont(QtGui.QFont('SansSerif', 10))
        #self.lab.setStyleSheet('color: green')

        self.lab1 = QLabel("\n\nArea under curve = " + str(auc),
                          self)
        # self.lab1.move(50, 30)
        self.lab1.setFont(QtGui.QFont('SansSerif', 10))
        # self.lab1.setStyleSheet('color: green')

        self.roc_button = QPushButton("View Roc Curve", self)
        #self.roc_button.setGeometry(QRect(375, 10, 111, 28))
        self.roc_button.clicked.connect(self.ROC)
        self.roc_button.setStyleSheet("QPushButton { background-color: #0e568a; color: white; }"
                                    "QPushButton:pressed { background-color: #000000; color:white }")

        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(2)
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setHorizontalHeaderLabels(['Positive(1)', 'Negative(0)'])
        self.tableWidget.setVerticalHeaderLabels(['Positive(1)', 'Negative(0)'])
        self.tableWidget.setItem(0, 0, QTableWidgetItem("TP = " + str(conf[0][0])))
        self.tableWidget.setItem(0, 1, QTableWidgetItem("FP = " + str(conf[0][1])))
        self.tableWidget.setItem(1, 0, QTableWidgetItem("FN = " + str(conf[1][0])))
        self.tableWidget.setItem(1, 1, QTableWidgetItem("TN = " + str(conf[1][1])))

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.lab)
        self.layout.addWidget(self.tableWidget)
        self.layout.addWidget(self.lab1)
        self.layout.addWidget(self.roc_button)
        self.setLayout(self.layout)

        self.show()

    def ROC(self):
        plt.plot(fpr, tpr, color='orange', label='ROC')
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.show()


class ThirdWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Overview")
        #self.setFixedSize(800, 600)
        self.setWindowIcon(QtGui.QIcon('icon1.png'))

        if (dataset_name != ""):
            df = pd.read_excel(dataset_name)
            yes = (df['Defective'] == 'Y').sum()
            no = (df['Defective'] == 'N').sum()

        self.graphical_view = QPushButton("Graphical View", self)
        # self.roc_button.setGeometry(QRect(375, 10, 111, 28))
        self.graphical_view.clicked.connect(self.graphical_overview)
        self.graphical_view.setStyleSheet("QPushButton { background-color: #0e568a; color: white; }"
                                          "QPushButton:pressed { background-color: #000000; color:white }")

        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(3)
        self.tableWidget.setColumnCount(2)
        self.tableWidget.horizontalHeader().setVisible(False)
        self.tableWidget.verticalHeader().setVisible(False)
        self.tableWidget.setItem(0, 0, QTableWidgetItem("Dataset Name"))
        self.tableWidget.setItem(0, 1, QTableWidgetItem(str(dataset_name)))
        self.tableWidget.setItem(1, 0, QTableWidgetItem("Instances"))
        self.tableWidget.setItem(1, 1, QTableWidgetItem(str(yes+no)))
        self.tableWidget.setItem(2, 0, QTableWidgetItem("Attributes"))
        self.tableWidget.setItem(2, 1, QTableWidgetItem(str(len(df.columns))))

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.tableWidget)
        self.layout.addWidget(self.graphical_view)
        self.setLayout(self.layout)
        self.show()

    def graphical_overview(self):
        df = pd.read_excel(dataset_name)
        yes = (df['Defective'] == 'Y').sum()
        no = (df['Defective'] == 'N').sum()
        labels = 'Defective', 'Non-Defective'
        sizes = [yes, no]
        colors = ['gold', 'lightskyblue']
        explode = (0.1, 0)  # explode 1st slice
        # Plot
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=140)
        plt.title("Graphical View of dataset " + str(dataset_name))
        plt.axis('equal')
        plt.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Create and display the splash screen
    splash_pix = QPixmap('pic.jpg')

    splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    splash.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
    splash.setEnabled(False)
    splash.setGeometry(200,150,650,400)

    # splash.setMask(splash_pix.mask())

    splash.show()
    splash.showMessage("", Qt.AlignTop | Qt.AlignCenter, Qt.black)



    # Simulate something that takes time
    time.sleep(2)

    window = Window()
    splash.finish(window)
    sys.exit(app.exec())

