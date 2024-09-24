<h1>HDB Resale Price Predictor 🏡 </h1>
<h2>By Den, Justin, Barry and Alice</h2>

# Introduction

This project is collaborative effort of four teammates, Den, Justin, Barry and Alice, during our General Assembly Data Analyst Immersive Bootcamp on September 2024!

# Modelling

We chose lightGBM as our model because of its train and test r2, accuracy and its run time. Our biggest factor is that its ability to not depend on ONE variable stands out the most. We required a model that is able to take into multiple consideration when predicting prices and that it does not rely on one significant variable.

# Streamlit Demo
Demo over here:
<a href="https://dj-bab-hdb-sales-predictor.streamlit.app">HDB sales predictor</a>

The predictor uses the input variable to determine the HDB resale pricing. Our agents at WOW! can input these different variables to predict pricing.
1. Town
2. Flat type
3. Lease commencedate date
4. Storey range
5. Floor Area (Sq Ft)

## Step by Step Guide

### Step 1
You will first arrive on this page when you click on the link above (awake the app if it mentioned that it is idle due to inactive)


### Step 2
Select the choices in the dropdown boxes (as indicated by arrows)


### Step 3
Click "Predict Resale Price" (as indicated by arrow) and result will be as shown below.

### Step 4
You may download the result as CSV file for your use!

