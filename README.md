# Eye Disease Classification

## Project Overview
This project is a Streamlit web application that uses a deep learning model to classify eye images into different disease categories. The model is trained on a custom dataset and deployed for real-time inference.

## Features
* Upload an eye image and get an instant disease classification.
* Powered by a pre-trained **ResNet18** model fine-tuned for this task.
* Built with **Streamlit** for a simple and interactive user interface.

## How to Run Locally
Follow these steps to set up and run the application on your machine.

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-link>
    cd <your-repo-link>
    ```
2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    # For Windows
    venv\Scripts\activate
    # For macOS/Linux
    source venv/bin/activate
    ```
3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

## Model Training
The training process is documented in the `Model_Training.ipynb` Jupyter Notebook. It includes steps for data loading, preprocessing, model architecture, training, and evaluation. The final trained model is saved as `best_model.pth`.

## License
This project is licensed under the MIT License.