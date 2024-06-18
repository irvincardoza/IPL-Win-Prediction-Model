
# IPL Match Prediction

This project is based on the IPL match prediction used in the second innings of a cricket match. The trained model takes into account factors such as current run rate (CRR), required run rate (RRR), runs, and balls remaining. The model is trained after going through two major datasets, both of which are available on [Kaggle](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020).

- The `iplproj` directory contains the Django files and the trained model in the form of `pipe.pkl`.
- The `ipl.py` file contains the code used to train the data and create the model.


# Project Structure
```
iplproj/
├── iplapp/
│   ├── migrations/
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── templates/
│   │   └── iplapp/
│   │       └── predict.html
│   ├── urls.py
│   └── views.py
├── iplproj/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── pipe.pkl
└── manage.py
```


## How to Run the Project

1. Clone the repository:
    ```bash
    git clone https://github.com/irvincardoza/IPL-Win-Prediction-Model.git
    ```

2. Navigate to the project directory:
    ```bash
    cd plproj
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Apply migrations:
    ```bash
    python manage.py migrate
    ```

5. Run the development server:
    ```bash
    python manage.py runserver
    ```

6. Open your web browser and go to `http://127.0.0.1:8000/` to use the application.

## Usage

- Fill in the form with the details of the match situation to get the prediction of winning probabilities for both teams.

