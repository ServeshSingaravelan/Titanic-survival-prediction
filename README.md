# Titanic Survival Prediction Backend

This project is a Titanic Survival Prediction Backend built using python and FastAPI. It predicts the survival chances of individuals aboard the Titanic based on specific passenger details using a machine learning model. The project includes the backend logic to handle predictions, which is containerized using Docker and Docker Compose.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Folder Structure](#folder-structure)

## Prerequisites

Make sure you have the following installed on your system:

- Python (v3.9 or higher)
- Docker
- Docker Compose

## Installation

First, clone the repository to your local computer

```bash
git clone https://mygit.th-deg.de/ainb_24_semper_fortis/titanic_web_service.git
cd titanic_web_service/backend
pip install -r requirements.txt
```

Running the Application
To run the application using Docker and Docker Compose, follow these steps:

Build the Docker image:

```bash
docker-compose build
```
Run the Docker container:

```bash
docker-compose up
```

The FastAPI application will be accessible at http://localhost:8000.

Folder Structure

```bash
titanic_modeL_service/
  models/
    knn_model.pkl
    logistic_regression_model.pkl
    random_forest_model.pkl
  w_train.csv
  w_test.csv
  predictornew.py
  task.py
  Dockerfile
  docker-compose.yml
  requirements.txt
  README.md
```  
models/: Contains the pre-trained machine learning models.
predictornew.py: Script for data preprocessing and model prediction.
task.py: Main FastAPI application file.
Dockerfile: Dockerfile for containerizing the application.
docker-compose.yml: Docker Compose file to run the application.
requirements.txt: Python dependencies required for the application.
README.md: Documentation for the backend application.

