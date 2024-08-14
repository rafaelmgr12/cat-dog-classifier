# Cat-Dog Classifier

![Cat-Dog Classifier](https://img.shields.io/badge/Status-Active-brightgreen)
![Python Version](https://img.shields.io/badge/Python-3.8-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Description

The **Cat-Dog Classifier** is a machine learning project designed to classify images of cats and dogs. Using a trained neural network, the project can identify whether a given image contains a cat or a dog. This repository includes all the necessary code to train the model, make predictions, and serve the application via a web interface.

## Features

- **Image Classification**: Upload an image of a cat or dog, and the model will tell you which one it is.
- **Model Training**: Scripts to train the neural network with new data.
- **Web Interface**: A simple interface to upload images and view prediction results.
- **Deploy with Docker**: Easy deployment using Docker.

## Project Structure

- **`app.py`**: Main application file. Serves the web interface and handles classification requests.
- **`Dockerfile`**: Docker file for building the image and deploying the application.
- **`models/`**: Contains the pre-trained model `model1_catsVSdogs.h5`.
- **`process_data.py`**: Script for data preprocessing, preparing it for training.
- **`src/`**: Directory containing the main scripts:
  - `main.py`: Entry point to run the application.
  - `preds.py`: Script to generate predictions using the trained model.
  - `training_model.py`: Script to train the model from input data.
- **`templates/`**: Contains the HTML file (`index.html`) for the web interface.

## Requirements

- Python 3.8+
- Docker
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/rafaelmgr12/cat-dog-classifier.git
   cd cat-dog-classifier
    ```
2. Create a virtual environment and install the dependencies:
   ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
   ```
3. Run the application:
  ```bash
    python app.py

  ```

## Using Docker

To run the application using Docker, follow these steps:

1. Build the Docker image:
    
    ```bash
    docker build -t cat-dog-classifier .
    ```
    
2. Run the container:
    
    ```bash
    docker run -p 5000:5000 cat-dog-classifier
    
    ```
    
3. Access the application via `http://localhost:5000`.

## Model Training

If you want to train the model from scratch:

1. Ensure you have the training data.
2. Run the training script:
    
    ```bash
    python src/training_model.py
    
    ```
  
3. The new model will be saved in the `models/` directory.

## Contributing

Contributions are welcome! Feel free to open issues or pull requests.

## License

This project is licensed under the MIT License.
