import pandas as pd
import os
from flask import Flask, render_template, jsonify
from sklearn.cluster import KMeans

app = Flask(__name__)

DATASET_FILE = "dataset.xlsx"  
DATA_FILE = "data.txt"  
PROGRESS_FILE = "progress.txt"  
CHUNK_SIZE = 20  

def generate_data_txt():
    """
    Lê as linhas do arquivo Excel (dataset.xlsx) de forma incremental e salva em data.txt.
    Reinicia a coleta quando o final do dataset for alcançado.
    """
    
    if not os.path.exists(PROGRESS_FILE):
        start_index = 0
    else:
        with open(PROGRESS_FILE, "r") as f:
            start_index = int(f.read().strip())

    df = pd.read_excel(DATASET_FILE, skiprows=1, names=["idSensor", "temperature"])

    if start_index >= len(df):
        start_index = 0

    df_chunk = df.iloc[start_index:start_index + CHUNK_SIZE]

    start_index += CHUNK_SIZE

    if start_index >= len(df):
        start_index = 0

    with open(PROGRESS_FILE, "w") as f:
        f.write(str(start_index))

    df_chunk.to_csv(DATA_FILE, index=False)

    return "Arquivo data.txt atualizado"


def run_cluster():
    df = pd.read_csv('data.txt')
    data = df['temperature'].copy()
    i = 1
    past_inertia = 0
    current_inertia = 0
    while True:
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(pd.DataFrame(data))
        print(kmeans.inertia_)
        current_inertia = kmeans.inertia_
        if abs(current_inertia - past_inertia) < 1:
            break
        past_inertia = kmeans.inertia_
        i+=1
        if i >= 20:
            break
    
    df['status'] = kmeans.labels_
    group = df.groupby('status').count()
    size = df.shape[0]
    
    for index, row in df.iterrows():
        if ((group.iloc[int(row['status'])]['idSensor'] / size)*100 <= 5):
            df.at[index, 'label'] = 1
        else:
            df.at[index, 'label'] = 0
    
    df[['idSensor', 'temperature', 'label']].to_csv('result.txt', index=False)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate_and_run")
def generate_and_run():
    generate_data_txt() 
    run_cluster()  
    df = pd.read_csv("result.txt")  
    return jsonify(df.to_dict(orient="records")) 

if __name__ == "__main__":
    app.run(debug=True)
