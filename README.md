# On People Analytics

![](img/project-banner.png)

---

# 1.0 The context
![](img/slide_01.png)

![](img/slide_02.png)

![](img/slide_03.png)

![](img/slide_04.png)

![](img/slide_05.png)

![](img/slide_06.png)

![](img/slide_07.png)

![](img/slide_08.png)

All the references are stated at the end of this README.

<br>

# 2.0 The problem
As mentioned, companies go through a lot of trouble when there is employee turnover. Although businesses know that this is a big problem that causes high losses, they don't really know how to tackle this problem. 

The C-Level might say: 

> *"We have all this data collected from all of our employees. We have their performance ratings, their Best Place To Work answers, etc. However, what do we do about it? How this data can help us prevent turnover? Who are the people that are most likely to leave? And what can we do to retain them? We can't afford to lose money anymore. "*

<br>

# 3.0 The solution
The designed solution is a web application that helps the HR team to have, among different information about the employees - who are most likely to leave the company, their identification numbers (`employee_number`).

The additional information range from department, education field, job satisfaction, job role, monthly rates to several others.

Once having this information, the team can download it and start acting on it.

![](img/web-app.gif)

Check it live at: https://people-analytics-bk.herokuapp.com/

**OBS: It may take a while to load the app, because I'm using the free tier of Heroku and in this tier app hibernate after 30 min of inactivity.**

<br>

## 3.1 What drove the solution

### 3.1.1 Exploratory Data Analysis
#### Descriptive Analysis

![](img/descr-analysis.png)

Key points:

- There are **all sorts of Ages** ranging from 18 to 60.
- The lowest **performance rating** is 3 and the highest is 4.
- There are **people that works really far from home**. The farthest people live 29 km from work.
- There are people that has been **working at the same company** for 40 years.

<br>

#### Hypothesis Map

This map to help us to decide which variables we need in order to validate the hypotheses.

![](img/mind-map.jpg)

#### Univariate Analysis

![](img/univariate-analysis-attrition.png)

As observed, there are much more people that stayed than left the company.

<br>

### 3.1.2 Hypothesis validation - Bivariate Analysis
#### Main Hypothesis

#### H1. People up to 40s tend to leave.

![](img/H1_age.png)

Although, up to 22 years old, comparing people who do tend to leave with who don't, the proportion of people who do tend to leave is large. In addition, it seems that people in young ages (up to 40s) tend to leave more than people in elder ages (40+). 

> Thus, the hypothesis is **TRUE**.


#### H3. People who live far from work tend to leave.

![](img/H3_distance_home.png)

Observing the plots, as the distance gets higher, between 12 and 28, the tendency for an employee to leave is higher.

> Thus, the hypothesis is **TRUE**.


#### H11. People who feel less involved with the job tend to leave more.

![](img/H11_job_involvement.png)

As observed, people who feel less involved with the job **don't tend to leave more**.

> Thus, the hypothesis is **FALSE**.


#### H14. People who have lower work life balance tend to leave more.

![](img/H14_work_life_balance.png)

As observed, people who have lower work life balance **tend to leave less**.

> Thus, the hypothesis is **FALSE**.

#### H18. People who are making more money tend not to leave.

![](img/H18_making_money.png)

As observed, people who are daily making more money tend to stay. As the median for hourly rate and monthly rate are quite similar.

> Thus, the hypothesis is **TRUE**.

<br>

#### Hypothesis summary

Several other hypotheses were outlined and validated.

| Hypothesis                                                                                   | Conclusion                                                                                                                                                          |
|----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| H1\. People up to 40s tend to leave\.                                                        | True                                                                                                                                                                |
| H2\. People that have higher degree of education tend to leave more\.                        | False                                                                                                                                                               |
| H3\. People who live far from work tend to leave\.                                           | True                                                                                                                                                                |
| H4\. Single people tend to leave more\.                                                      | True                                                                                                                                                                |
| H5\. People who make overtime tend to leave more\.                                           | True                                                                                                                                                                |
| H6\. People who present higher performance ratings tend to leave more\.                      | False                                                                                                                                                               |
| H7\. People who present lower performance ratings tend to leave more\.                       | True                                                                                                                                                                |
| H8\. People who have lower job level tend to leave more\.                                    | True                                                                                                                                                                |
| H9\. People who weren't promoted for long time tend to leave more\.                          | False                                                                                                                                                               |
| H10\. People who are in the current role for long time tend to leave more\.                  | False                                                                                                                                                               |
| H11\. People who feel less involved with the job tend to leave more\.                        | False                                                                                                                                                               |
| H12\. People who feel less satisfied with the job tend to leave more\.                       | False                                                                                                                                                               |
| H13\. People who feel less satisfied with the environment tend to leave more\.               | We can't really say if people who feel less satisfied with the environment tend to leave more, because the counts are almost equal for each level of satisfaction\. |
| H14\. People who have lower work life balance tend to leave more\.                           | False                                                                                                                                                               |
| H15\. People who professionally worked for more years tend to not leave\.                    | True                                                                                                                                                                |
| H16\. People who worked at the same company for more years tend not to leave\.               | True                                                                                                                                                                |
| H17\. People who are job hoppers tend to leave more\.                                        | False                                                                                                                                                               |
| H18\. People who are making more money tend not to leave\.                                   | True                                                                                                                                                                |
| H19\. People who have shorter salary hike range tend to leave\.                              | True                                                                                                                                                                |
| H20\. People who received less training last year tend to leave more\.                       | People who received few and many training sessions last year tend to stay\. However, people who are in the middle tend to leave\.                                   |
| H21\. People who have been working for the same manager for short years tend to leave more\. | True                                                                                                                                                                |
| H22\. People who have lower quality of relationship with the manager tend to leave more\.    | False                                                                                                                                                               |
| H23\. People who travel more frequently tend to leave more\.                                 | False                                                                                                                                                               |
| H24\. Which departments has more turnover?                                                   | As observed, Research & Development has more turnovers than other departments\.                                                                                     |
| H25\. Which education field has more turnover?                                               | Life Sciences is the education field which has more turnover, followed by Medical and Marketing\.                                                                   |

<br>

### 3.1.3 Machine Learning

Tests were made using different algorithms.

#### Performance Metrics 
![](img/ml_alg_comparison.png)


#### PR Curve
![](img/ml_alg_pr_auc.png)


#### Confusion Matrix
![](img/ml_alg_cm.png)

<br>

#### 3.1.3.1 Conclusion

As we are dealing with **imbalaced data set**, the most relevant metric, in this case, is the **F1-Score** which is used **when the False Negatives and False Positives are crucial**. For this project, False Negatives are crucial, since losing these people could lead to company financial loss. Thus, the algorithm that best suits the needs is **Balanced Random Forest Classifier**.

<br>

### 3.1.4 Business Performance

As mentioned early, when an employee leaves the company, the position will have to be replaced which leads to a high cost and energy consuming hiring process (head hunting, CV review, interviews, tests, onboarding, etc.)

So, when building a machine learning model we are going to focus on optimizing its performance metrics, that is, **minimizing the False Negatives while maximizing the True Positives**.

<img src="img/confusion-matrix.png" width="70%">

In addition, we can outline a **best-worst scenario** for an employee who leaves a company.

|                                  | Best scenario | Worst scenario |
|----------------------------------|---------------|----------------|
| **Cost \($\)**                   | 4,000         | 7,645          |
| **Time to fill a position \(days\)** | 42        | 52             |

Testing the model using a data set containing 368 records, it was able to correctly identify 42 (True Positives) and miss 17 (the False Negatives) from a total of 59 employees who tend to leave. Translating to the best-worst scenario, this means:

|                                  | Best scenario | Worst scenario |
|----------------------------------|---------------|----------------|
| **Total loss prevented \($\)**   | 168,000       |     321,090    |
| **Total time saved \(days\)**    | 1,764         | 2,184          |


In addition, **without the model**, the company would have 59 employees that could leave, translating it to **a total loss of \$ 236,000 in the best scenario and \$ 451,055 in the worst scenario**.

<br>

### 3.1.5 Machine Learning Performance for the chosen algorithm

The chosen algorithm was the **Balanced Random Forest**.

#### Precision and Recall, AUC and Confusion Matrix

| Accuracy | F1-Score | PR AUC |
|----------|----------|--------|
| 0.779891 | 0.509091 | 0.405141 |

<br>

|              | precision | recall | f1\-score | support |
|-------------:|----------:|-------:|----------:|--------:|
| 0            | 0\.94     | 0\.79  | 0\.86     | 309     |
| 1            | 0\.40     | 0\.71  | 0\.51     | 59      |
| accuracy     |           |        | 0\.78     | 368     |
| macro avg    | 0\.67     | 0\.75  | 0\.68     | 368     |
| weighted avg | 0\.85     | 0\.78  | 0\.80     | 368     |

![](img/model-pr-auc-cm.png)

<br>

#### Probability Distribution

![](img/model-pb-dist.png)

Here, we are evaluating how normally distributed are the probabilities predicted by the model.

- **Histogram of predicted probabilities:** it nearly follows a normal distribution. The normal distribution is depicted in black-line bell shaped curve.

- **Probability Plot:** when can observe that almost all the points are or on the red diagonal line or close to it, which means that the probability distribution is near normal.

- **ECDF plot:** the cumulative distribution function almost follows the half of a normal distribution, which means that the probability distribution is near normal.

All these 3 elements show that the model is able to make good predictions, of course, it can always be improved.

<br>

# 4.0 Next Steps

- Build an **analytics dashboard** in a data visualization tool (e.g. Tableau, Power BI) so the HR team can have a clear view about the characteristics of employees that are most likely to leave the company.

- **Recommend decisions** to the HR team based on groups of employees so it can make the right initiatives to prevent turnover.

- **Test other techniques** to train the model, including artificial neural networks (e.g. TensorFlow, PyTorch).

- **Include Design Thinking applied to Employee Experience aided by Data Science**, because only identifying the employees is not enough if companies don't know how to approach them. Actually, this can be the part two of this project which I described in [this Medium post](https://medium.com/@brunokatekawa/on-people-analytics-employee-turnover-b493cec75f17).

<br>

# References

https://business.linkedin.com/talent-solutions/blog/employee-retention/2019/gallup-suggests-employee-turnover-in-us-business-is-1-trillion-dollar-problem-with-simple-fix

https://www.gallup.com/workplace/247391/fixable-problem-costs-businesses-trillion.aspx

https://www.gallup.com/workplace/260564/heard-quit-rate-win-war-talent.aspx

https://toggl.com/blog/cost-of-hiring-an-employee

