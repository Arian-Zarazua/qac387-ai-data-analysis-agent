# Build 0 – Data Analysis Pipeline (Assignment 1)

**Course:** QAC387  
**Assignment Type:** Individual  
**Focus:** Python functions, error checks, reproducible analysis pipelines  

---

## Assignment Overview

In this assignment, you will complete **Build 0** of a data analysis pipeline.  

The goal is to understand how **small, well-defined Python functions** can be combined into a complete, reproducible workflow.

You are given a working pipeline with a few **intentional gaps** in which you must complete the code. Your task is to fill in those gaps **without breaking the rest of the pipeline**.

You will also build two new functions from scratch.

You will then run the code from the terminal. The code should run **without error** and **populate the reports folder** with output.

The functions in this Build0 data analysis pipeline assignment will serve as tools in future agentic workflow.

---

### Learning Objectives

By completing this assignment, you should be able to:

1) Write simple Python functions with clear inputs and outputs

2) Use pandas to summarize missing data

3) Add defensive checks using ValueError

4) Understand how functions fit into a larger pipeline

5) Run a Python script with command-line arguments

### Repository Structure

```text
.
├── builds/
│   ├── build0_data_analysis_pipeline_assignment_1.py   # YOU EDIT THIS FILE
├── data/
│   ├── penguins.csv   # THE DATA SET YOU WILL USE TO TEST YOUR PIPELINE CODE
│   
├── requirements.txt
└── README.md
```

### Set Up

1. Clone the repository called ```qac387-ai-data-analysis-agent``` in Prof Rose's Github account (jrose01) to a local Git repository

2. Create and activate your virtual environment, then run:

```pip install -r requirements.txt```

This will install all the packages and dependencies that you will need to run the code.

### What You Need to Do

Step 1) Fill in the blanks in the code with the correct code.

Step 2) Build the following 2 functions (hints are provided in the code):

**Function 1**:

```python
missingness_table(df)
```

The purpose of this function will be to create a table that summarizes missing data by column.

Required output columns include:

a) the target variable column

b) missing_rate (proportion missing, between 0 and 1)

c) missing_count (number of missing values)

*Table Requirements:*

Returns a pandas data frame that has one row for each column in the penguins dataframe that is sorted by missing_rate in descending order (most frequent missing data to least frequent missing data)

**Function 2**:

```python
multiple_linear_regression(df, outcome, predictors=None)
```

The purpose of this function wil be to perform a multiple linear regression analysis with a numeric outcome variable.

*Requirements:*

If the outcome variable is not numeric, raise:

```python
ValueError("Outcome must be numeric for linear regression: <outcome>")
```

If predictors is ```None``` then:

Use all other columns as predictors

Missing data:

Drop rows with missing values (listwise deletion)

**Output:**

The function must return a string summary suitable for saving to a .txt file

## How to Run the Pipeline

Run the script from the terminal command line using the commands below:

1) Basic run (no regression):

```bash
python builds/build0_data_analysis_pipeline_assignment_1.py \
--data path/to/penguins.csv
```

2) Run with regression:

```bash
python builds/build0_data_analysis_pipeline_assignment_1.py \
  --data path/to/penguins.csv \
  --reg_outcome OutcomeColumnName
```

### Expected Output

After a successful run, a reports/ folder will be created containing:

reports/
├── data_profile.json
|
├── summary_numeric.csv
|
├── summary_categorical.csv
|
├── missingness_by_column.csv
|
├── correlations.csv
|
├── regression_summary.txt
|
└── figures/
    ├── missingness.png
    |
    ├── corr_heatmap.png
    |
    ├── hist_<column>.png
    |
    └── bar_<column>.png

### Final Step

1) Update your requirements.txt file by typing the following command in the terminal:
```bash
pip freeze > requirements.txt
```
2) Push your local repo to your remote Github account:

a) **Commit** your changes
b) click **Publish Repository** (you do this the first time to create a remote Github respository for your local repository)
     - save as a private repo
     - do not add a README file
     - do not add a .gitignore
c) **Push** your local repo with changes to your remote Github repo
d) add Profs Rose and Kim and the TAs as **Collaborators** on your Github repo

### Common Pitfalls

1) Returning the wrong column names in missingness_table

2) Forgetting to sort by missing_rate

3) Not raising an error for categorical regression outcomes

4) Returning something other than a string from regression

5) Modifying code outside the TODO sections

### Advice

1) Read the docstrings carefully — they are part of the assignment

2) Run the script before submitting

3) Review any error messages — they are designed to help you
