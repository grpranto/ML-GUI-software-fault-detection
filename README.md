<a name="readme-top"></a>
<p float="left">
  <img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue">
  <img src="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white">
  <img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white">
  <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white">
  <img src="https://img.shields.io/badge/Qt-41CD52?style=for-the-badge&logo=qt&logoColor=white">
</p>

<!-- ABOUT THE PROJECT -->
### About The Project
This machine learning project is a part of my Bachelor thesis and [publication](https://link.springer.com/chapter/10.1007/978-3-030-33709-4_5) with the goal of checking the effect of feature selection methods in software fault detection. The GUI saved a lot of time during our research as it is capable of generating any combinations of results

The GUI consists of: 
* five classifiers: `Decision Tree`, `Random Forest`, `Naïve Bayes`, `Logistic Regression`, `Neural Network`
* five feature selection techniques: `Information Gain`, `Relief`, `Chi-Square`, `Chi-square Test of Independence`, `Feature Importance`
* five datasets: NASA’s benchmark publicly available datasets

Result includes: `accuracy`, `confusion matrix`, `AUC`, `ROC curve`

[![ML-GUI-SoftwareFaultDetection](https://github.com/grpranto/ML-GUI-software-fault-detection/blob/main/screenshots/interface1.PNG?raw=true)](abcd)

The GUI can be used in multiple ways. We can choose any of the classifiers without feature selection and check the result. We can also choose classifiers along with feature selection technique. For example, at fist step, we can only take the `Logistic Regression` as classifier and see the result. At second step, we can take the `Logistic Regression` as classifier along with the feature selection technique `Information gain` and produce the result. After comparing the results, we can easily reach to a conclusion if there is any effect of the feature selection technique in software fault detection.


### What is the advantage? 
Total combinations possible = 5 classifiers * 5 feature selection techniques * 5 datasets = 125

Just select any combination from the GUI and get the desired result :)

* Compare the result of `classifier alone` vs `classifer with feature selection technique`
* User Friendly
* Save a lot of time
* Don't have to maintain different python/jupyter notebook files for different combinations

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Appendix (More Screenshots of the tool):
#### Dataset Overview
![ML-GUI-SoftwareFaultDetection](https://github.com/grpranto/ML-GUI-software-fault-detection/blob/main/screenshots/interface2.PNG?raw=true) 

![ML-GUI-SoftwareFaultDetection](https://github.com/grpranto/ML-GUI-software-fault-detection/blob/main/screenshots/interface3.PNG?raw=true)

#### Result of an arbitrary combination(`Logistic Regression` along with `Information gain`)
![ML-GUI-SoftwareFaultDetection](https://github.com/grpranto/ML-GUI-software-fault-detection/blob/main/screenshots/interface4.PNG?raw=true) 

![ML-GUI-SoftwareFaultDetection](https://github.com/grpranto/ML-GUI-software-fault-detection/blob/main/screenshots/interface5.PNG?raw=true)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
<!--https://github.com/alexandresanlim/Badges4-README.md-Profile-->
