import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import json




data_dict = json.load(open(r"C:\Users\schum\Downloads\stl_metadata.json"))


def DBSCAN_outlier_detection(data_dict: dict, epsilon: float = 0.07, min_samples: int = 20):
    df = pd.DataFrame.from_dict(data_dict, orient='index').reset_index()
    df.rename(columns={'index': 'obj_id'}, inplace=True)
    features = ['Mesh_volume_mm3', 'Surface_Area_mm2']
    X = df[features].values


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(X_scaled)
    df['cluster'] = db.labels_
    df['status'] = df['cluster'].apply(lambda x: 'Outlier' if x == -1 else f'Cluster {x}')
    print(df[['obj_id', 'Mesh_volume_mm3', 'status']], )

    fig = px.scatter(
        df, 
        x="Mesh_volume_mm3", 
        y="Surface_Area_mm2", 
        color="status",
        hover_name="obj_id",
        hover_data={"obj_id": False, "Mesh_volume_mm3": ':.3f', "Surface_Area_mm2": ':.3f', 'status': True}
        , # This puts the ID label next to the point
        title=f"STL Outlier Detection using DBSCAN with epsilon {epsilon} and min_samples {min_samples}",
        labels={"Mesh_volume_mm3": "Volume [mm^3]", "Surface_Area_mm2": "Surface Area [mm^2]"},
        template="plotly_dark",
        color_discrete_map={'Outlier': 'orange'} # Highlight outliers in red
    )

    fig.update_traces(textposition='top center')
    fig.show() # This opens the interactive plot in your browser
    fig.write_html("segmentation_report.html")


if __name__ == "__main__":
    DBSCAN_outlier_detection(data_dict, epsilon=0.07, min_samples=20)   