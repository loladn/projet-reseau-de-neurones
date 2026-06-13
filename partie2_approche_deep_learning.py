import os
import re
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import random
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ==========================================
# 1. CONFIGURATION
# ==========================================

NOMS_LABELS = [
    'Jungle', 'Plage', 'Monuments', 'Bus', 'Dinosaures', 
    'Elephants', 'Fleurs', 'Chevaux', 'Montagne', 'Plats'
]
DICO_LABELS = dict(zip(range(10), NOMS_LABELS))

# Chemin vers le dossier d'images
REPERTOIRE_IMAGES = os.path.join('data', 'Wang')

# Taille des images pour le CNN (réduit à 128x128 pour économiser la mémoire)
TAILLE_IMAGE = (128, 128) 

# ==========================================
# 2. DATA ENGINEERING (Chargement Images)
# ==========================================

def charger_images_et_labels(repertoire, taille_image):
    """
    Charge les images .jpg, les redimensionne et génère les labels
    basés sur le nom du fichier (ex: 100.jpg -> classe 1).
    Inspire du code tutorat.
    """
    if not os.path.exists(repertoire):
        raise FileNotFoundError(f"Le dossier {repertoire} n'existe pas.")

    print(f"Chargement des images depuis {repertoire}...")
    images = []
    labels = []
    
    # On trie pour avoir un ordre déterministe (0.jpg, 1.jpg...)
    fichiers = sorted(os.listdir(repertoire), key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else 0)

    count = 0
    for nom_fichier in fichiers:
        # Regex pour vérifier format "chiffre.jpg"
        if nom_fichier.endswith('.jpg') and re.match(r'^\d+\.jpg$', nom_fichier):
            try:
                chemin = os.path.join(repertoire, nom_fichier)
                
                # Chargement et Redimensionnement
                with Image.open(chemin) as img:
                    img = img.convert('RGB') # Assurer 3 canaux (Couleur)
                    img_redim = img.resize(taille_image)
                    
                    # Normalisation (0-255 -> 0-1) avec float32 pour économiser la mémoire
                    img_array = np.array(img_redim, dtype=np.float32) / 255.0
                    images.append(img_array)
                    
                    # Label : 0-99 -> 0, 100-199 -> 1 ...
                    numero = int(nom_fichier.split('.')[0])
                    etiquette = numero // 100
                    labels.append(etiquette)
                    count += 1
            except Exception as e:
                print(f"Erreur image {nom_fichier}: {e}")
    
    print(f"{count} images chargées avec succès.")
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)

def preparer_donnees(images, labels):
    """Divise en Train/Val/Test et encode les labels."""
    # Encodage One-Hot
    labels_cat = to_categorical(labels, num_classes=10)
    
    # Split Train (70%) / Temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels_cat, train_size=0.7, stratify=labels, random_state=42
    )
    
    # Split Val (15%) / Test (15%) -> Donc 50% du Temp chacun
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# ==========================================
# 3. MODELISATION (CNN)
# ==========================================

def creer_modele_pdf(forme_entree, nombre_classes):
    """
    Architecture simple suggérée par le sujet (PDF).
    Conv2D -> Relu -> MaxPool -> Flatten -> Dense -> Softmax
    """
    model = Sequential([
        # Couche de convolution : extraction de features (bords, couleurs)
        Conv2D(32, (3, 3), input_shape=forme_entree),
        Activation('relu'),
        
        # Pooling : réduction de la taille (garde l'info importante)
        MaxPooling2D(pool_size=(2, 2)),
        
        # Aplatissement pour passer au classifieur
        Flatten(),
        
        # Classification (Full Connected)
        Dense(nombre_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def creer_modele_avance(forme_entree, nombre_classes):
    """
    Architecture améliorée (optimisée pour faible mémoire).
    Moins de filtres pour réduire la consommation mémoire.
    """
    model = Sequential([
        # Bloc 1
        Conv2D(16, (3, 3), padding='same', input_shape=forme_entree, activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        
        # Bloc 2
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        # Bloc 3 (réduit)
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(nombre_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ==========================================
# 4. VISUALISATION
# ==========================================

def tracer_courbes(history, titre=""):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Train Acc')
    plt.plot(epochs, val_acc, 'r--', label='Val Acc')
    plt.title(f'{titre} - Précision')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Train Loss')
    plt.plot(epochs, val_loss, 'r--', label='Val Loss')
    plt.title(f'{titre} - Perte')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def afficher_matrice(y_true, y_pred):
    classes = list(DICO_LABELS.values())
    cm = confusion_matrix(y_true, y_pred)
    z_text = [[str(y) for y in x] for x in cm]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=classes, y=classes, text=z_text, texttemplate="%{text}",
        colorscale='Viridis'
    ))
    fig.update_layout(title='Matrice de Confusion CNN', width=700, height=600)
    fig.show()

# ==========================================
# 5. EXECUTION PRINCIPALE
# ==========================================

if __name__ == "__main__":
    
    # 1. Chargement
    try:
        images, labels = charger_images_et_labels(REPERTOIRE_IMAGES, TAILLE_IMAGE)
    except FileNotFoundError as e:
        print(e)
        exit()
        
    if len(images) == 0:
        print("Aucune image trouvée. Vérifiez le chemin 'data/Wang'.")
        exit()

    # 2. Préparation
    print("\n--- Préparation des données ---")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preparer_donnees(images, labels)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    input_shape = (TAILLE_IMAGE[0], TAILLE_IMAGE[1], 3)
    num_classes = 10
    
    # 3. Entraînement Modèle Simple (Consigne PDF)
    print("\n--- Entraînement Modèle PDF (Simple) ---")
    model_simple = creer_modele_pdf(input_shape, num_classes)
    # model_simple.summary()
    
    callback_simple = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    
    hist_simple = model_simple.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=20, batch_size=16, callbacks=callback_simple, verbose=1
    )
    
    score_simple = model_simple.evaluate(X_test, y_test, verbose=0)
    print(f"Accuracy Modèle Simple : {score_simple[1]*100:.2f}%")
    tracer_courbes(hist_simple, "Modèle Simple PDF")

    # 4. Entraînement Modèle Avancé (Pour comparaison)
    print("\n--- Entraînement Modèle Avancé (Optimisé) ---")
    model_opti = creer_modele_avance(input_shape, num_classes)
    
    callbacks_opti = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=3)
    ]
    
    hist_opti = model_opti.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=30, batch_size=8, callbacks=callbacks_opti, verbose=1
    )
    
    score_opti = model_opti.evaluate(X_test, y_test, verbose=0)
    print(f"Accuracy Modèle Avancé : {score_opti[1]*100:.2f}%")
    tracer_courbes(hist_opti, "Modèle Avancé")

    # 5. Évaluation finale (sur le meilleur modèle)
    best_model = model_opti if score_opti[1] > score_simple[1] else model_simple
    print(f"\n--- Analyse détaillée du meilleur modèle ({'Avancé' if best_model == model_opti else 'Simple'}) ---")
    
    y_pred_prob = best_model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    afficher_matrice(y_true, y_pred)
    print(classification_report(y_true, y_pred, target_names=NOMS_LABELS))
    
    # 6. Test visuel sur image inconnue
    print("\n--- Test Visuel ---")
    indices = random.sample(range(len(X_test)), 3)
    for i in indices:
        img = X_test[i]
        true_label = NOMS_LABELS[np.argmax(y_test[i])]
        
        pred_prob = best_model.predict(img.reshape(1, TAILLE_IMAGE[0], TAILLE_IMAGE[1], 3), verbose=0)
        pred_label = NOMS_LABELS[np.argmax(pred_prob)]
        conf = np.max(pred_prob)
        
        plt.figure(figsize=(2,2))
        plt.imshow(img)
        plt.title(f"Vrai: {true_label}\nPred: {pred_label} ({conf:.2f})")
        plt.axis('off')
        plt.show()