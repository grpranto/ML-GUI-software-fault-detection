<a name="readme-top"></a>
<!-- ABOUT THE PROJECT -->
### About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

This project is a part of my Bachelor thesis and [publication](https://link.springer.com/chapter/10.1007/978-3-030-33709-4_5) with the goal of checking the effect of feature selection methods in software fault detection. The experiment discovers that the most important and relevant features can be selected by the adopted feature selection techniques without sacrificing the performance of fault detection.

The GUI consists of: 
* five classifiers: `Decision Tree, Random Forest, Naïve Bayes, Logistic Regression, Neural Network`
* five feature selection techniques: `Information Gain, Relief, Chi-Square, Chi-square Test of Independence, Feature Importance`
* five datasets: NASA’s benchmark publicly available datasets

We can choose any of the combinations of the mentioned classifiers and feature selection technique. For example, we can take the `Logistic Regression` as classifier and `Information gain` as feature selection technique. Then we can compute the result only for that combination. Result includes calculating accuracy, computing confusion matrix, generating ROC curve, etc. 


### What is the advantage? 
Total combinations possible = 5 classifiers * 5 feature selection techniques * datasets = 125

Just select any combination from the GUI and get the desired result :)

* User Friendly
* Save a lot of time
* Don't have to maintain different python/jupyter notebook files for different combinations

### Technology/Language:
Python, PyQT, Pandas, Numpy, Scikit-Learn, Matplotlib


<p align="right">(<a href="#readme-top">back to top</a>)</p>
