### **BirdSong Clustering**

- **Level 9 - Overview :**  

  This project analyzes, and predicts wildlife sounds to classify them and show their similarity based on their audios.
  There's also the "bird_sound.py" file, which show how I made these short samples.
  
  *Small note : I didn’t specify birdsong clearly enough in the request, so I ended up getting general wildlife sounds instead. 
  But honestly, that works out well since I wanted to make things more challenging anyway.*


  It uses unstructured data : 

    - 400 audios from the Xeno Canto's API : [Xeno Canto Website](https://xeno-canto.org/)
    - Live fetching pictures from INaturalist's API : [INaturalist Website](https://www.inaturalist.org/)
  
  It includes : 

    - Requesting data (Requests, Xeno Canto's API),
    - Fetching pictures data (Requests, INaturalist's API),
    - Feature Engineering (Librosa : cleaning, sampling and extracting infos from sounds; 
      Pandas : creating dataset for the model),
    - Unsupervised learning algorithm (K-Means, silhouette metric), 
    - Dimension reduction (PCA),
    - Machine learning pipelines (with RandomizedSearchCV), 
    - Interactive graph (Plotly clustering 3D scatter graph),
    - Random audios (Streamlit audio display),
    - Distance calculation for approximative closest points (Scipy Spatial),
    - Dashboards and more (coming soon).

  Main purposes (in WIP) : 

     - to identify similarity between species.
     - to analyze birds from the same specie with different wildlife sounds. 
     - to check the hypothesis of "similar audio, similar environment"?  

  
  There is only version available:
  - 🇬🇧 **English version**: Includes English commentary (`-en`).

  - **Tech Stack:** Python, Pandas, Streamlit, Scikit-learn, Librosa, Requests.


---

⚠️ Reminder: This is a beginner-level project : a test, an exploration, a digital sketchbook.         
⚠️ Note: This app might take a few seconds to load if it's been inactive — please be patient while it wakes up!

---

### **How to Run the Project:**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/gen-x13/Level-9
   ```
---

### **Demo**

👉 [Click here to test the Streamlit live app](https://birdsongsandmusics.streamlit.app/)

---

### **Requirements**  
Before running the project, make sure you have the following libraries installed:  
```bash
pip install pandas streamlit plotly scikit-learn librosa pathlib plotly requests

```
---

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](./LICENSE) file for details.

---

### **💜 A Reminder:**

***Trust the process, your pace and progress.***
*Each one of us has its own melody.*

> ### Link to the bilingual blog talking about this project (french & english):
> ### [Blog Clustering BirdSongs Project | Clustering Chant d'oiseaux Projet Link](https://ko-fi.com/post/Clustering-BirdSongs-Project-Clustering-Chant-d-I3I61U135S)
