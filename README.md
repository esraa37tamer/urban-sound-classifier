# 🎶 Urban Sound Classifier  

A **cutting-edge deep learning project** that blends  
🎵 *machine listening* with 🎨 *interactive design*.  

Built with **Python** + **Streamlit**, this app classifies **urban environmental sounds**  
(car horns, sirens, drilling, street music, and more) in real time,  
with an elegant, intuitive web interface.  

---

## ✨ Highlights
- 🎧 **Upload & Play** — Drop in your audio file, preview it instantly.  
- 🤖 **Smart Predictions** — Powered by deep learning (CNNs).  
- 📊 **Confidence Scores** — See probabilities for each sound class.  
- 🎨 **Responsive UI** — Dark/Light theme friendly, clean, minimal design.  
- ⚡ **One-Click App** — Just run, no configs needed.  

---

## 🛠️ Tech Stack
- **Python 3.9+**  
- [Streamlit](https://streamlit.io/) → Modern web interface  
- [TensorFlow / Keras](https://www.tensorflow.org/) → Deep learning backbone  
- [Librosa](https://librosa.org/) → Audio feature extraction (MFCCs, spectrograms)  
- [NumPy & Pandas](https://pandas.pydata.org/) → Efficient data handling  
- [Matplotlib & Seaborn](https://matplotlib.org/) → Visualization  

---

## ⚙️ Quickstart  

Clone the repository and set up the environment:

```bash
git clone https://github.com/esraa37tamer/urban-sound-classifier.git
cd urban-sound-classifier
pip install -r requirements.txt

---

## 📂 Project Structure
urban-sound-classifier/
│
├── es.py                 # Main Streamlit app
├── requirements.txt      # Dependencies
├── models/               # Pre-trained model(s)
├── data/                 # Dataset / sample audios
├── utils/                # Helper functions (feature extraction, etc.)
├── assets/               # Images, icons
└── README.md             # Documentation
