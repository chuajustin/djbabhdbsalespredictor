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
![Step 1](https://github.com/user-attachments/assets/65a7c4d2-f87e-4858-a3e5-9c2a3f8fb4a9)

### Step 2
Select the choices in the dropdown boxes (as indicated by arrows)  
![Step 2](https://github.com/user-attachments/assets/6d433888-b6f0-4934-9b39-97e29ad4c1ab)


### Step 3
Click "Predict Resale Price" (as indicated by arrow) and result will be as shown below.  
![Step 3](https://github.com/user-attachments/assets/1312a555-c5ef-4d98-8d67-93dd932e5267)
![Step 4](https://github.com/user-attachments/assets/4b5242fb-87fc-4376-88d5-eb79ade6e9c7)


### Step 4
You may download the result as CSV file for your use!  
![Step 5](https://github.com/user-attachments/assets/4445ce12-e54d-4347-a04b-0ece4ac53a35)

