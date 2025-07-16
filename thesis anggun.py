import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ======================= RULES DEFINISI ==========================
rules = [
    ({"G001", "G002", "G003", "G004", "G011", "G012"}, "Maag Ringan"),
    ({"G001", "G002", "G003", "G004", "G006", "G007", "G010", "G011", "G012"}, "Maag Sedang"),
    ({"G001", "G002", "G003", "G004", "G011", "G012", "G006"}, "Maag Sedang"),
    ({"G001", "G002", "G003", "G004", "G011", "G012", "G007"}, "Maag Sedang"),
    ({"G001", "G002", "G003", "G004", "G011", "G012", "G010"}, "Maag Sedang"),
    ({"G001", "G002", "G003", "G004", "G011", "G012", "G006", "G007"}, "Maag Sedang"),
    ({"G001", "G002", "G003", "G004", "G011", "G012", "G006", "G010"}, "Maag Sedang"),
    ({"G001", "G002", "G003", "G004", "G011", "G012", "G007", "G010"}, "Maag Sedang"),
    ({"G001", "G002", "G003", "G004", "G005", "G006", "G007", "G008", "G009", "G010", "G011d", "G012"}, "Maag Kronis"),
    ({"G001", "G002", "G003", "G004", "G011", "G012", "G006", "G007", "G010", "G005"}, "Maag Kronis"),
    ({"G001", "G002", "G003", "G004", "G011", "G012", "G006", "G007", "G010", "G008"}, "Maag Kronis"),
    ({"G001", "G002", "G003", "G004", "G011", "G012", "G006", "G007", "G010", "G009"}, "Maag Kronis"),
    ({"G001", "G002", "G003", "G004", "G011", "G012", "G006", "G007", "G010", "G005", "G008"}, "Maag Kronis"),
    ({"G001", "G002", "G003", "G004", "G011", "G012", "G006", "G007", "G010", "G005", "G009"}, "Maag Kronis"),
    ({"G001", "G002", "G003", "G004", "G011", "G012", "G006", "G007", "G010", "G008", "G009"}, "Maag Kronis")
]

# =================== FUNGSI FORWARD CHAINING =====================
def forward_chaining(rules, initial_facts):
    facts = set(initial_facts)
    diagnosis = None
    new_facts = True
    while new_facts:
        new_facts = False
        for conditions, result in rules:
            if conditions.issubset(facts) and result not in facts:
                facts.add(result)
                diagnosis = result
                new_facts = True
    return diagnosis

# ======================= MENU ============================
mode = input("PILIH MENU:\n1. Diagnosa \n2. Evaluasi Performa \nPilihan (1/2): ").strip()

if mode == "1":
    print("\n=== PREDIKSI ===")
    print("Masukkan kode gejala (pisahkan dengan koma), contoh: G001,G002,G003")
    input_gejala = input("Gejala: ")
    initial_facts = set(g.strip().upper() for g in input_gejala.split(","))
    
    start_time = time.time()
    hasil_diagnosa = forward_chaining(rules, initial_facts)
    end_time = time.time()

    print("\n=== Hasil Diagnosa ===")
    print(f"Gejala yang dimasukkan: {initial_facts}")
    print(f"Diagnosa: {hasil_diagnosa if hasil_diagnosa else 'Bukan Maag'}")
    print(f"Waktu Eksekusi: {end_time - start_time:.6f} detik")

else:
    print("\n=== PERFORMA ===")

    # ================== DATA EVALUASI ==========================
    y_true = np.array([1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 2, 1, 2, 1, 2, 2, 2, 1, 1, 1, 2, 2, 1, 2, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 0, 0, 2, 1, 2, 2, 2, 1, 1, 1, 2, 2, 1, 2, 0, 0, 0, 0, 0, 2])

    # ================== CONFUSION MATRIX =======================
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Bukan Maag", "Maag Ringan", "Maag Sedang"],
                yticklabels=["Bukan Maag", "Maag Ringan", "Maag Sedang"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - Diagnosa Maag")
    plt.show()

    # ================== KONVERSI BINER ==========================
    y_true_binary = np.array([1 if x in [1, 2] else 0 for x in y_true])
    y_pred_binary = np.array([1 if x in [1, 2] else 0 for x in y_pred])
    cm_binary = confusion_matrix(y_true_binary, y_pred_binary)
    TN, FP, FN, TP = cm_binary.ravel()

    print("\nConfusion Matrix (Biner):\n", cm_binary)
    print(f"True Positive (TP): {TP}")
    print(f"False Positive (FP): {FP}")
    print(f"True Negative (TN): {TN}")
    print(f"False Negative (FN): {FN}")

    accuracy_binary = accuracy_score(y_true_binary, y_pred_binary)
    precision_binary = precision_score(y_true_binary, y_pred_binary)
    recall_binary = recall_score(y_true_binary, y_pred_binary)
    f1_binary = f1_score(y_true_binary, y_pred_binary)

    print("\n=== Metrik Evaluasi  ===")
    print(f"Akurasi  : {accuracy_binary:.2f}")
    print(f"Presisi  : {precision_binary:.2f}")
    print(f"Recall   : {recall_binary:.2f}")
    print(f"F1-Score : {f1_binary:.2f}")
