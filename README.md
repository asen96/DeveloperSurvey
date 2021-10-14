# DeveloperSurvey

## Predicting Salary using Stack Overflow Developer Survey Results

StackOverflow is a popular forum for software developers, students, and anyone who deals with programming, in a context for professional work or education. The website conducts annual surveys of its users, which gathers data about respondents in terms of their employment status, work experience, technology usage, along with personal details, such as age, gender, sexuality etc. This data is published online and can be accessed by anyone freely. It provides valuable insights into the workers in the technology sector, as well as people who wish to be employed in this sector in the future or have been in the past. 

One interesting way of using the StackOverflow survey data is to use it to train a neural network which would predict the salary of an individual, given certain input parameters. The features used to train the model are responses from the survey. In this project, the data from 2019 was used, since it had the largest number of respondents who provided their annual salary. 

## Describing the data

The csv files containing the survey results is read using pandas and converted to a Data Frames, for each year. For 2019, the data frame has 88,883 rows (responses) and 84 columns (features), including the annual salary. Here are a couple of plots visualizing the data from 2017 to 2021, depending on the availability of certain features.

### Age
This bar plot shows the distribution of ages of the respondents from 2018 to 2021. The survey results from 2017 did not include the age of the respondents.

![](/visualizations/age_distribution.svg)

### Salary
This plot shows the distribution of annual salaries in US$ of the respondents.

![](/visualizations/comp_distribution.svg)

### Education
This chart shows the highest educational qualifications of the respondents from 2017 to 2021. In every year, respondents with a bachelor's degree make up a clear plurality of respondents, followed by respondents with master's degrees and with no formal qualifications.

![](./visualizations/education_distribution.svg)

This plot shows the average salaries of respondents corresponding to particular educational qualifications. A clear increase in average salary can be seen while going from respondents with no formal educational qualifications to respondents with bachelor's degrees (+11.71%), and also going from master's degrees and doctorate degrees (+25.53%).

![](./visualizations/salary_by_education.svg)

### Coding Experience

The chart below shows the distribution of respondents with years of experience in professional coding (left), along with the average salary in each of those groups of respondents (right)

![](./visualizations/experience_and_salary.svg)

## Predicting the salary

### Preprocessing (preprocessing.py)

Preprocessing the data involved the following main steps:
1. Dividing the columns into three categories : numerical, multiple, and exclusive. Numerical columns contain responses which are numerical in nature, for eg. number of hours of work per week, age (after some filtering), and the salary. Multiple columns are the ones whose elements contain multiple responses to the question, for eg. programming languages used. Exclusive columns are ones whose elements contain mutually exclusive responses, for eg. country, employment status.
2. The multiple and exclusive columns were exploded into one-hot-encoded columns, for each response. This makes it easier to pass through a neural network. 
3. Outliers are identified as respondents whose responses (especially the numerical ones), fall beyond the 3\*sigma interval of the mean of the responses. They also are respondents who have not provided any data regarding their salary. The columns are then scaled by dividing by the upper limit 3\*sigma.


