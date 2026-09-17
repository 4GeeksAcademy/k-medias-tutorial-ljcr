from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
FEATURES = ["Latitude", "Longitude", "MedInc"]
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "housing.csv"


def main():
	housing = pd.read_csv(DATA_PATH, usecols=FEATURES)
	train_df, test_df = train_test_split(
		housing,
		test_size=0.20,
		random_state=RANDOM_STATE,
	)
	train_df = train_df.copy()
	test_df = test_df.copy()

	kmeans = KMeans(n_clusters=6, random_state=RANDOM_STATE, n_init=10)
	train_df["cluster"] = pd.Categorical(kmeans.fit_predict(train_df[FEATURES]))

	sns.set_theme(style="whitegrid")
	plt.figure(figsize=(10, 7))
	sns.scatterplot(
		data=train_df,
		x="Longitude",
		y="Latitude",
		hue="cluster",
		palette="tab10",
		s=18,
		alpha=0.65,
	)
	plt.title("Clusters K-Means: conjunto de entrenamiento")
	plt.tight_layout()
	plt.savefig(PROJECT_ROOT / "train_clusters.png", dpi=150)
	plt.close()
	print(
		"Los clusters forman regiones geograficas diferenciadas a lo largo de "
		"California, con mayor concentracion de observaciones en las zonas costeras."
	)

	test_df["cluster"] = pd.Categorical(kmeans.predict(test_df[FEATURES]))

	figure, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
	sns.scatterplot(
		data=train_df,
		x="Longitude",
		y="Latitude",
		hue="cluster",
		palette="tab10",
		s=15,
		alpha=0.30,
		legend=False,
		ax=axes[0],
	)
	axes[0].set_title("Entrenamiento")
	sns.scatterplot(
		data=test_df,
		x="Longitude",
		y="Latitude",
		hue="cluster",
		palette="tab10",
		s=20,
		alpha=0.80,
		ax=axes[1],
	)
	axes[1].set_title("Prueba: clusters predichos")
	figure.suptitle("Comparacion espacial de clusters")
	figure.tight_layout()
	figure.savefig(PROJECT_ROOT / "train_test_clusters.png", dpi=150)
	plt.close(figure)

	classifier = RandomForestClassifier(
		n_estimators=200,
		random_state=RANDOM_STATE,
		n_jobs=-1,
	)
	classifier.fit(train_df[FEATURES], train_df["cluster"])
	predicted_clusters = classifier.predict(test_df[FEATURES])
	print("\nReporte de clasificacion:\n")
	print(classification_report(test_df["cluster"], predicted_clusters))
	joblib.dump(kmeans, PROJECT_ROOT / "kmeans_model.pkl")
	joblib.dump(classifier, PROJECT_ROOT / "clf_model.pkl")
	print("Modelos guardados como kmeans_model.pkl y clf_model.pkl")


if __name__ == "__main__":
	main()
