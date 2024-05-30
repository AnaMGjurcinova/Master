import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score



# Učitavanje podataka

df = pd.read_excel(r"C:\Users\agjur\Desktop\AnaMarija.xlsx")

# Cilj je da podelimo klijente na različite rangove za svaku od tri promenljive: Recency (R), Frequency (F), i Monetary (M).
# Ovaj proces pomaže da se identifikuju klijenti koji su ostvarili više transakcija (veća vrednost promenljive Frequency)
# sa većim iznosima (manja vrednost promenljiive Monetary) u kraćim vremenskim intervalima (manja vrednost promenljiive Recency).

# Prvo, koristimo funkciju pd.qcut da podelimo vrednosti svake promenljive u kvantile.
# Ova funkcija automatski sortira vrednosti pre nego što ih podeli, što osigurava da svaki kvantil ima približno jednak broj klijenata.
# Na primer, ako imamo 100 klijenata i delimo ih u 5 kvantila, svaki će kvantil imati oko 20 klijenata.

# Zatim, nakon što su klijenti podeljeni u kvantile za svaku promenljivu, dodeljujemo im rangove.
# Klijentima koji su više i češće trošili dodeljujemo veći rang za promenljive Recency (R), Frequency (F), i Monetary (M) respektivno.

df['R_Rank'] = pd.qcut(df['Recency'], q=5, labels=[5,4, 3, 2, 1])
df['F_Rank'] = pd.qcut(df['Frequency'], q=5, labels=[1, 2, 3, 4,5])
df['M_Rank'] = pd.qcut(df['Monetary'], q=5, labels=[5,4,3,2,1])

# Računamo finalnu RFM promenljivu kao prosek gorenavedenih rangova.

df['RFM_Score'] = (df['R_Rank'].astype(int) + df['F_Rank'].astype(int) + df['M_Rank'].astype(int))/3
# Zamena autlajera za godine klijenta

Q1 = df['god_klijenta'].quantile(0.25)
Q3 = df['god_klijenta'].quantile(0.75)
IQR = Q3 - Q1
#print(IQR, Q1 , Q3)

autlajeri = (df['god_klijenta'] < (Q1 - 1.5 * IQR)) | (df['god_klijenta'] > (Q3 + 1.5 * IQR))
# Zamenjivanje autlajera srednjom vrednošću
df_clean = df.copy()
df_clean['god_klijenta'] = df['god_klijenta'].mask(autlajeri, df['god_klijenta'].mean())
# Delimo klijente u 7 grupa na osnovu godina
df_clean['age_group'] = np.where(df_clean['god_klijenta'] <= 25, 1,
                            np.where((df_clean['god_klijenta'] > 25) & (df_clean['god_klijenta'] <= 35), 2,
                            np.where((df_clean['god_klijenta'] > 35) & (df_clean['god_klijenta'] <= 40), 3,
                            np.where((df_clean['god_klijenta'] > 40) & (df_clean['god_klijenta'] <= 45), 4,
                            np.where((df_clean['god_klijenta'] > 45) & (df_clean['god_klijenta'] <= 55), 5,
                            np.where((df_clean['god_klijenta'] > 55) & (df_clean['god_klijenta'] <= 65), 6,
                             7))))))

# Pripremamo uzorak za klasterovanja na osnovu RFM promenljive i godina

data = df_clean[['RFM_Score', 'age_group']]

# Histogram gustine

plt.hist(df_clean['god_klijenta'], density=True)
plt.xlabel('Godine')
plt.ylabel('Gustina verovatnoće')
plt.title('Histogram')
plt.show()

# Lista za čuvanje vrednosti Siluet indeksa
silhouette_scores = []

# Pokušajte različite brojeve klastera i izračunajte Siluet indeks
for i in range(2, 11):
    model = AgglomerativeClustering(n_clusters=i, linkage='average')  # Može biti bilo koja metoda klasterovanja
    labels2 = model.fit_predict(data)
    silhouette_scores.append(silhouette_score(data, labels2))

# Prikazivanje rezultata Siluet indeksa
plt.plot(range(2, 11), silhouette_scores)
plt.title('Koeficijent siluete za hijerarhijsko klasterovanje')
plt.xlabel('Broj klastera')
plt.ylabel('Koeficijent siluete')
plt.show()

# Primena metode hijerahijskog klasterovanja

# Vardova metoda
hierarchical_cluster_a = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')
hierarchical_cluster_a.fit(data)
labels_a = hierarchical_cluster_a.labels_
# Metoda potpunog vezivanja
hierarchical_cluster_b = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='complete')
hierarchical_cluster_b.fit(data)
labels_b = hierarchical_cluster_b.labels_
# Metoda jednim vezom
hierarchical_cluster_c = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='single')
hierarchical_cluster_c.fit(data)
labels_c = hierarchical_cluster_c.labels_

# Metoda prosečnog vezivanja
hierarchical_cluster_d = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='average')
hierarchical_cluster_d.fit(data)
labels_d = hierarchical_cluster_d.labels_


# Tabelarni prikaz klastera

cluster_counts = np.bincount(labels_a)
print("Broj klijenata u svakom klasteru:")
for cluster_id, count in enumerate(cluster_counts):
    print(f"Klaster {cluster_id}: {count} klijenata")

df_clean['Cluster'] = hierarchical_cluster_d.labels_ # Možemo da zamenimo a, b, c ili d u zavisnosti sa kojom metodom radimo

# Statistike klastera
cluster_stats = df_clean.groupby('Cluster').agg({'RFM_Score': ['min', 'max'],
                                                 'god_klijenta': ['min', 'max'],
                                                 'Flag_LastM': 'sum',
                                                 'Flag_Last3M' : 'sum',
                                                 'Cluster' : 'count'
                                                })


print(cluster_stats)


# Pretpostavimo da imamo dva klasterovanja dobijena različitim metodama
# Računanje Rand indeksa
rand_index = adjusted_rand_score(labels_a, labels_b)
print("Rand indeks:", rand_index)
