Violence Detection System
This project aims to detect violent activities in video streams or uploaded videos using deep learning models. It features a complete end-to-end system with real-time alerts, a responsive frontend, and backend database storage.

📁 Project Structure
backend/: Python backend using Flask and FastAPI.
frontend/: React-based user interface.
documents/: Contains the demo video, presentation (PPT), and documentation file.
violence.db: SQLite database for storing detection results.
.h5 model: Trained deep learning model for violence detection.

🛠️ Environmental Setup
The development environment is configured with the following libraries:

opencv-python – For video capture and frame processing.
Flask – Lightweight web backend for APIs and alert handling.
FastAPI – Efficient RESTful API handling.
pandas – Data preprocessing and analysis.
numpy – Mathematical operations on arrays and frames.
seaborn – Visualization during evaluation/debugging.

🧠 Dataset Used
Kaggle: Real-Life Violence Situations Dataset
This dataset was used to train the .h5 model used in the system.

Configured on a system with:
Intel Core i7 processor
16 GB RAM

⚠️ Note: Since requirements.txt is not provided in the repo, you'll need to install the packages manually or create the file using pip freeze > requirements.txt after setting up your environment.

📦 Installation Steps
Clone the repository:

bash

git clone https://github.com/Niharika6767/ViolenceDetectionSystem.git
cd ViolenceDetectionSystem
Set up Python virtual environment (recommended):

bash

python -m venv venv
venv\Scripts\activate  # Windows
Install dependencies manually:

bash

pip install opencv-python flask fastapi pandas numpy seaborn
Start backend:

bash

cd backend
python app.py
Start frontend:

bash

cd frontend
npm install
npm start
🧠 Model & Database
Model: A pre-trained .h5 model is used to detect violence in video frames.

Database: An SQLite database violence.db is used to store detection results.

Table: results_data

Fields: id, filename, timestamp, violence (0 or 1), frame_path

📹 Live Stream Setup
For real-time detection:

Install the IP Webcam app on your mobile device.
Connect the device and PC to the same network.
Enter the IP webcam URL in the app’s live stream input field on the frontend.

Start detection.

🔐 User Features
User registration and login
Upload video or live stream detection
Real-time violence alert system

View detection results and analytics

📂 Supporting Files
Demo video, documentation, and PPT available in the documents/ folder.
