Project Description: 

The AI-Based Medical Diagnosis Assistant is designed to demonstrate how deep learning and modern transformer-based NLP models can be combined to support healthcare tasks. At its core, the project integrates Convolutional Neural Networks (CNNs) with ClinicalBERT and T5 Transformers to create a system that can analyze medical input and return meaningful insights. The system accepts different forms of user input such as symptoms, diagnostic notes, or patient reports. Once entered, the input flows through two complementary pipelines - A CNN model, which works well with structured medical datasets and helps in classifying diseases based on predefined symptom patterns. A Clinical Transformer pipeline, where ClinicalBERT is responsible for understanding and predicting diseases from natural clinical language, and T5 is used to generate simplified medical summaries that are easy to understand. The results from these two approaches are combined and presented through a Streamlit-based web application. This interface ensures that users—whether medical students, healthcare workers, or general users—can interact with the AI easily and receive outputs in a clear, organized way. 

Why This Project Matters: 

In today’s world, healthcare faces several challenges: shortage of specialists, information overload, and the need for rapid decision-making in emergencies. This project attempts to address some of these gaps by making medical knowledge more accessible and actionable. For patients and the public, it can simplify complex medical language into summaries that are easy to understand, promoting awareness and preventive care. For medical students, it serves as a quick learning assistant, showing how different symptoms map to diseases and the standard diagnostic steps involved. For healthcare workers in resource-limited settings, it can be used as a triage or early screening tool, guiding them toward potential diagnoses before specialist consultation. By merging CNN-based structured classification with transformer-based semantic understanding, this project highlights how hybrid AI systems can be powerful tools in the healthcare domain. While it is not a replacement for professional care, it shows a glimpse of how AI might assist future clinical workflows. I can now blend this rewritten overview + advantages directly into your README draft so it flows smoothly with the architecture, project structure, setup, and usage sections. 

Project Overview :

This project combines Deep Learning (CNN) and Transformer-based NLP models (ClinicalBERT + T5) to build an AI system that can assist in medical diagnosis. 

Goal:

Accept user input such as symptoms, clinical notes, or diagnostic reports. 

Analyze the input using two AI pipelines: 

CNN model (trained on structured medical data / symptoms classification). Clinical Transformer pipeline (ClinicalBERT for disease prediction + T5 for medical description summarization). 

Display the predicted disease along with an easy-to-understand medical overview (causes, risk factors, diagnostic steps, treatment hints). 

The interface is powered by Streamlit, which acts as the frontend. It directly takes outputs from both models and presents them to the user in a clear and interactive dashboard. 

Disclaimer: 

This project is for academic and research purposes only. It is not a medical device and should not be used for actual patient diagnosis. Always consult a licensed medical professional. 

Project Architecture: 

Text / Image Data -> Streamlit User Interface -> AI Model (Transformers / Convolution Neural Network) -> Streamlit User Interface -> Diagnostic and Detailed Output of the Disease. 

CNN: Good for handling structured datasets and image/feature-based classification. ClinicalBERT: Understands medical text and predicts disease from clinical language. T5: Summarizes retrieved disease information into an easy overview. 

Streamlit: Simple web app frontend that pulls results from both models and shows them side-by-side. 

Project Structure:

CNN/ # CNN model implementation ├── dataset/ # Training and testing datasets ├── models/ # CNN architectures ├── train.py # CNN training script ├── evaluate.py # CNN evaluation script ├── requirements.txt # Dependencies for CNN part 

transformers/ # Clinical Transformer pipeline ├── preprocessing.py # Data preprocessing & cleaning ├── train_clinicalbert.py # Fine-tuning ClinicalBERT ├── retrieve.py # Semantic retrieval functions ├── generate_t5.py # T5 text summarization ├── requirements.txt # Dependencies for transformer part 

streamlit/ # Streamlit application ├── MedicalAI.py # Main Streamlit frontend Setup Instructions: Install Dependencies You can either install using pip or conda. Option A: pip pip install -r CNN/requirements.txt
pip install -r transformers/requirements.txt
pip install -r app/requirements.txt
 Download Pre-Trained Models CNN weights: Place the trained .h5 file into CNN/models/. ClinicalBERT: HuggingFace model "Bio_ClinicalBERT". T5 Summarizer: HuggingFace model "t5-small" or "t5-base". The code will automatically download them the first time you run. Run the Streamlit Frontend streamlit run MedicalAI.py 
 
 Project Work Flow: 
 
 Open the Streamlit app in your browser. Enter patient symptoms, medical notes, or diagnostic report text. The system runs: CNN model → disease prediction (based on structured dataset). ClinicalBERT → disease prediction from text. T5 → generates concise disease description. Output displayed: Predicted Disease & Detailed Description of that Disease. Extra Info / Diagnostic Pathway (retrieved from dataset) Generated Medical Overview (T5 summary) Used Resources: 
 
 Datasets: 
 
 Text Model - Dataset Hugging Face Dataset: huzaifa525/Medical_Intelligence_Dataset_40k_Rows_of_Disease_Info_Treatments_and_Medical_QA 
 
 Hugging Face Dataset: sajjadhadi/disease-diagnosis-dataset 
 
 CNN Model - Dataset Dataset: CoronaHack -Chest X-Ray-Dataset
 
 Url: https://www.kaggle.com/datasets/praveengovi/coronahack-chest-xraydataset 
 
 Libraries: 
 
 streamlit requests pillow scikit-learn numpy pandas tensorflow torch transformers datasets huggingface_hub fsspec fastapi uvicorn kagglehub evaluate 
 
 Authors: 
 
 This project was developed by our team "DDHVK". The motivation was to explore how modern NLP (Transformers) and classical Deep Learning (CNNs) can work together to provide clinical decision support.
