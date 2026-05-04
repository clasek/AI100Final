# AI100Final
Final project repository for Penn State AI 100 based on classification if the New York Jets win or lose a football game. The goal of this repository is to create and identify bug cases for the system in which it intentionally breaks. It will then get explained and debugged by giving a prompt to a LLM.

The dataset is automatically generated when the program runs, so no CSV file needs to be manually included in the repository.

The model is implemented in Python using:

PyTorch (neural network)

Pandas (data handling)

Scikit-learn (train/test splitting)

AI_100/ ├── src/ │ ├── generate_data.py # Generates the dataset │ └── AI100FINAL.py # Trains and evaluates the model ├── data/ # Empty folder; dataset will be generated here ├── README.md ├── .gitignore

The dataset consists of 600 simulated games. Each game includes:

points_scored

points_allowed

total_yards

turnovers

home (0 = away, 1 = home)

winner (0 = loss, 1 = win)

The winner is determined programmatically in generate_data.py.

If the dataset does not exist, it is automatically generated when running the main script.

HOW TO RUN

Clone the repository git clone https://github.com/clasek/AI100Midterm1.git cd AI100Midterm1
Install dependencies pip install pandas torch scikit-learn
Run the project python src/midtermcode.py
If jets_games.csv does not exist, it will be created automatically.

Training runs for 100 epochs.

Training loss and validation loss are displayed.

Final test accuracy is printed at the end.

Author: Christian Lasek
