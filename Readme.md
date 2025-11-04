# 🚦 Traffic Sign Detection Using AI

---

## 🧠 About the Project

### **Project Title:**  
**Traffic Sign Detection Using AI**

---

### **Overview**
This project focuses on developing an **Artificial Intelligence–based system** that automatically detects and recognizes **traffic signs** from road images.  
The main goal is to assist in building **smart driving systems** and **enhance road safety** by allowing computers to identify traffic signs such as **Speed Limit**, **No Entry**, **U-Turn**, and **Pedestrian Crossing**.  

This system showcases how **AI can transform transportation** by enabling automated recognition of road rules and enhancing driver assistance systems.

---

### **Objective**
To build a **deep learning model** capable of detecting and classifying different traffic signs in static road images using **object detection techniques**.

---

### **🧩 Technology Stack**

- **Programming Language:** Python  
- **Framework:** YOLO (You Only Look Once) Object Detection  
- **Libraries:** OpenCV, NumPy, Pandas, Pillow, Matplotlib  
- **Environment:** Anaconda / Conda Virtual Environment  

---

### **🌍 Real-World Use Cases**
- In **self-driving cars**, such systems help vehicles automatically understand and follow road signs like Speed Limits or No Entry.  
- **Traffic surveillance systems** can detect and record sign violations through CCTV feeds.  
- **Navigation and map applications** can alert drivers about nearby road signs or hazards in real time.  
- **Smart city analytics** can integrate this model to monitor compliance and improve road safety infrastructure.  
- **Transport authorities** can use this model to identify accident-prone zones where certain signs are frequently ignored.  

---

## ⚙️ Environment Setup

Before running the project, create and activate a virtual environment (you can name it anything), then install dependencies:

```bash
# Create and activate a virtual environment
conda create -n <env_name> python=3.10 -y
conda activate <env_name>

# Install required libraries
pip install -r Requirements.txt
```

---

## 🧠 How This Project Works (In Simple Words)

1. You give the system a **road image** (for example, a picture containing traffic signs).  
2. The AI model (trained using **YOLO**) looks at the image and **searches for any traffic signs** it recognizes.  
3. It then **draws colored boxes** around the detected signs and **labels them** (like “Speed Limit 40” or “No Entry”).  
4. The **final image**, with all detected signs and confidence scores, is automatically **saved** in the output folder.  

So basically:  
> 🖼️ Input → Image with traffic signs  
> 🤖 Processing → AI model detects and classifies each sign  
> 📤 Output → Image with bounding boxes and labels showing detected signs  

---

### **⚙️ How It Works (Technical Steps)**

1. A **YOLO-based model** is trained on a labeled dataset of traffic signs.  
2. The trained model is tested on real-world road images using `test_images.py`.  
3. The script processes each image, detects the traffic signs, and displays **bounding boxes** with class labels and confidence scores.  
4. The output images are saved in `runs/detect/test_results_clean/`.

---

### **📊 Results**
The model detects multiple types of traffic signs with reasonable accuracy.  
However, due to hardware limitations (like lack of GPU support), it may not detect every sign perfectly.  
Still, it successfully demonstrates the working of an **AI-based road sign detection system**.

---

### **🚀 Future Scope**

- Improve accuracy by training on **larger and more diverse datasets**.  
- Implement **real-time detection** using live video input.  
- Integrate with **embedded systems (like Raspberry Pi)** for on-road deployment.  

---

## 🧾 How to Run the Project

### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/Traffic-Sign-Detection-using-AI.git
cd "Traffic Sign Detection using AI"
```

### **2. Install Dependencies**
Make sure you have Python and Conda installed.  
Then install the required libraries:
```bash
pip install -r Requirements.txt
```

### **3. Add Test Images**
Place your road images (with traffic signs) inside:
```
data/test/
```

### **4. Run the Detection Script**
Execute:
```bash
python test_images.py
```

### **5. View Results**
Annotated detection results will be saved automatically in:
```
runs/detect/test_results_clean/
```

---

## 📂 Dataset Source
The dataset used in this project was downloaded from **Kaggle**:  
🔗 [Traffic Signs Dataset (Indian Roads)](https://www.kaggle.com/datasets/kaustubhrastogi17/traffic-signs-dataset-indian-roads)

It contains labeled images of **Indian traffic signs**, used for both **training and testing** the AI model for detection.

---

### 📁 Folder Structure
```
D:\Data Science Projects\Traffic Sign Detection using AI
│
├── data/
│   ├── test/
│   │   ├── test 1.jpg
│   │   ├── test 12.jpg
│   │
│   ├── train/
│   │   ├── images/
│   │   │   ├── IMG_5257.jpeg
│   │   │   ├── IMG_5897.jpg
│   │   │
│   │   ├── labels/
│   │   │   ├── IMG_5257.txt
│   │   │   ├── IMG_5897.txt
│   │   │
│   │   └── labels.cache
│   │
│   ├── val/
│   │   ├── images/
│   │   │   ├── IMG_5260.jpg
│   │   │   ├── IMG_5895.jpg
│   │   │
│   │   ├── labels/
│   │   │   ├── IMG_5260.txt
│   │   │   ├── IMG_5895.txt
│   │   │
│   │   └── labels.cache
│
├── Dataset/
│   ├── images/
│   │   ├── IMG_5257.jpeg
│   │   ├── IMG_5897.jpg
│   │
│   ├── labels/
│   │   ├── IMG_5257.txt
│   │   ├── IMG_5897.txt
│
├── runs/
│   ├── detect/
│   │   ├── test_results_clean/
│   │   │   ├── test 1.jpg
│   │   │   ├── test 12.jpg
│   │   │
│   │   ├── train/
│   │   │   ├── weights/
│   │   │   └── args.yaml
│   │   │
│   │   ├── train2/
│   │   │   ├── weights/
│   │   │   └── args.yaml
│   │   │
│   │   ├── train3/
│   │   │   ├── weights/
│   │   │   └── args.yaml
│   │   │
│   │   ├── train4/
│   │   │   ├── weights/
│   │   │   │   ├── best.pt
│   │   │   │   └── last.pt
│   │   │   │
│   │   │   ├── args.yaml
│   │   │   ├── BoxF1_curve.png
│   │   │   ├── BoxP_curve.png
│   │   │   ├── BoxPR_curve.png
│   │   │   ├── BoxR_curve.png
│   │   │   ├── confusion_matrix.png
│   │   │   ├── confusion_matrix_normalized.png
│   │   │   ├── labels.jpg
│   │   │   ├── results.csv
│   │   │   ├── results.png
│   │   │   ├── train_batch0.jpg
│   │   │   ├── train_batch1.jpg
│   │   │   ├── train_batch2.jpg
│   │   │   ├── val_batch0_labels.jpg
│   │   │   ├── val_batch0_pred.jpg
│   │   │   ├── val_batch1_labels.jpg
│   │   │   ├── val_batch1_pred.jpg
│   │   │   ├── val_batch2_labels.jpg
│   │   │   └── val_batch2_pred.jpg
│
├── Screenshots/
│   │   ├── detection result 1.jpg
│   │   ├── detection result 2.jpg
│   │   ├── detection result 3.jpg
│   │   ├── detection result 4.jpg
│   │   ├── detection result 5.jpg
│   │   ├── detection result 6.jpg
│   │   ├── detection result 7.jpg
│   │   ├── detection result 8.jpg
│   │   ├── detection result 9.jpg
│   │   ├── detection result 10.jpg
│   │   ├── detection result 11.jpg
│   │   └── detection result 12.jpg
│
├── data.yaml
├── Readme.md
├── Requirements.txt
├── split_data.py
├── test_images.py
├── train_model.py
├── .gitignore
└── yolov8s.pt
```

---

## 🧠 Author
**Developed by:** Sohum Patil  
**Field:** Data Science and Artificial Intelligence  
**Goal:** To contribute toward intelligent transportation and road safety through AI.

---

### 📬 Feedback
💌 For suggestions or collaboration:  
**sohum7even@gmail.com**

---

⭐ *If you like this project, don’t forget to star the repository on GitHub!*
