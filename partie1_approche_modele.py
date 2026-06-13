import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import xlrd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ==========================================
# 1. CONFIGURATION ET NOMS DE CLASSES
# ==========================================

NOMS_LABELS = [
    'Jungle', 'Plage', 'Monuments', 'Bus', 'Dinosaures', 
    'Elephants', 'Fleurs', 'Chevaux', 'Montagne', 'Plats'
]
DICO_LABELS = dict(zip(range(10), NOMS_LABELS))

CHEMIN_FICHIER = os.path.join('data', 'WangSignatures.xls')

# ==========================================
# 2. DATA ENGINEERING (Chargement & Préparation)
# ==========================================

def charger_donnees_excel(chemin_fichier):
    """
    Charge les descripteurs depuis le fichier Excel WangSignatures.xls.
    """
    if not os.path.exists(chemin_fichier):
        raise FileNotFoundError(f"Le fichier n'a pas été trouvé à l'emplacement : {chemin_fichier}")

    print(f"Chargement du fichier : {chemin_fichier}...")
    
    try:
        df = pd.read_excel(chemin_fichier, header=None)
    except ImportError:
        print("Erreur: Il manque une librairie. Essayez : pip install xlrd openpyxl")
        return None, None

    # Les descripteurs (X) sont toutes les colonnes SAUF la première (qui est le nom de l'image)
    X = df.iloc[:, 1:].values 
    
    # Génération des labels (y) : 100 images par classe
    y = np.array([i // 100 for i in range(len(df))])
    
    print(f"Données chargées. Dimension X: {X.shape}, Dimension y: {y.shape}")
    return X, y

def preparer_donnees(X, y, test_size=0.15, val_size=0.15):
    """
    Divise, Normalise et Encode les données.
    """
    # 1. Division Train+Val / Test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    
    # 2. Division Train / Validation
    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_size, stratify=y_temp, random_state=42
    )
    
    # 3. Normalisation (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Encodage One-Hot
    num_classes = len(np.unique(y))
    y_train_enc = to_categorical(y_train, num_classes)
    y_val_enc = to_categorical(y_val, num_classes)
    y_test_enc = to_categorical(y_test, num_classes)
    
    return (X_train_scaled, y_train_enc), (X_val_scaled, y_val_enc), (X_test_scaled, y_test_enc)

# ==========================================
# 3. MODELISATION (MLP)
# ==========================================

def creer_modele_mlp(input_shape, num_classes):
    """
    Crée le modèle principal (MLP).
    """
    model = Sequential()
    
    # Couche d'entrée + Première couche cachée
    model.add(Dense(256, activation='relu', input_shape=(input_shape,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Deuxième couche cachée
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Couche de sortie
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compilation
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ==========================================
# 4. FONCTIONS DE VISUALISATION & ANALYSE
# ==========================================

def tracer_courbes(history):
    """Trace les courbes d'apprentissage (Loss et Accuracy)."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Train Accuracy')
    plt.plot(epochs, val_acc, 'r--', label='Validation Accuracy')
    plt.title('Précision du modèle')
    plt.xlabel('Epoques')
    plt.ylabel('Précision')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Train Loss')
    plt.plot(epochs, val_loss, 'r--', label='Validation Loss')
    plt.title('Fonction de coût (Loss)')
    plt.xlabel('Epoques')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def afficher_matrice_confusion(y_true, y_pred_classes):
    """Affiche la matrice de confusion avec Plotly."""
    classes = list(DICO_LABELS.values())
    cm = confusion_matrix(y_true, y_pred_classes)
    z_text = [[str(y) for y in x] for x in cm]

    fig = go.Figure(data=go.Heatmap(
        z=cm, x=classes, y=classes, text=z_text, texttemplate="%{text}",
        colorscale='Blues', colorbar=dict(title='Nombre d\'images')
    ))

    fig.update_layout(
        title='Matrice de Confusion - Jeu de Test',
        xaxis_title='Classe Prédite',
        yaxis_title='Classe Réelle',
        width=700, height=600
    )
    fig.show()

def systeme_discrimination(vecteur_mesures, modele, noms_classes):
    """Simule le système final : Prédit la classe d'une image donnée."""
    vecteur_redimensionne = vecteur_mesures.reshape(1, -1)
    prediction_probas = modele.predict(vecteur_redimensionne, verbose=0)
    index_classe = np.argmax(prediction_probas)
    return noms_classes[index_classe], np.max(prediction_probas)

def comparer_hyperparametres(X_train, y_train, X_val, y_val, X_test, y_test):
    """Entraîne plusieurs architectures pour comparer les performances."""
    configs = [
        {'nom': 'Modele_Simple', 'couches': [64], 'dropout': 0.0},
        {'nom': 'Modele_Moyen_Reg', 'couches': [128, 64], 'dropout': 0.3},
        {'nom': 'Modele_Gros_Dropout', 'couches': [512, 256, 128], 'dropout': 0.5},
        {'nom': 'Modele_Profond_Sans_Dropout', 'couches': [256, 128, 64, 32], 'dropout': 0.0}
    ]
    
    resultats = []
    print("\n--- DÉBUT DE LA COMPARAISON DES HYPERPARAMÈTRES ---")
    input_dim = X_train.shape[1]
    num_classes = y_train.shape[1]
    
    for conf in configs:
        print(f"\nEntraînement configuration : {conf['nom']} ...")
        model = Sequential()
        model.add(Dense(conf['couches'][0], activation='relu', input_shape=(input_dim,)))
        if conf['dropout'] > 0: model.add(Dropout(conf['dropout']))
        
        for neurones in conf['couches'][1:]:
            model.add(Dense(neurones, activation='relu'))
            if conf['dropout'] > 0: model.add(Dropout(conf['dropout']))
            
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32, callbacks=[es], verbose=0)
        
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f" -> Résultat : Accuracy = {acc*100:.2f}%")
        
        resultats.append({
            'Configuration': conf['nom'], 'Architecture': str(conf['couches']),
            'Dropout': conf['dropout'], 'Accuracy_Test': acc
        })
        
    res_df = pd.DataFrame(resultats).sort_values(by='Accuracy_Test', ascending=False)
    print("\n--- CLASSEMENT DES MODÈLES ---")
    print(res_df)
    
    plt.figure(figsize=(10, 6))
    plt.barh(res_df['Configuration'], res_df['Accuracy_Test'], color='skyblue')
    plt.xlabel('Précision (Accuracy)')
    plt.title('Comparaison des performances selon les hyperparamètres')
    plt.grid(axis='x')
    plt.show()

# ==========================================
# 5. EXECUTION PRINCIPALE
# ==========================================

if __name__ == "__main__":
    
    # 1. Chargement
    try:
        X, y = charger_donnees_excel(CHEMIN_FICHIER)
    except FileNotFoundError as e:
        print(e)
        exit()
        
    if X is not None:
        # 2. Préparation des données
        print("\n--- Préparation des données (Split & Normalisation) ---")
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = preparer_donnees(X, y)
        
        print(f"Train : {X_train.shape[0]} images")
        print(f"Val   : {X_val.shape[0]} images")
        print(f"Test  : {X_test.shape[0]} images")
        
        # 3. Création du modèle principal
        input_dim = X_train.shape[1]
        num_classes = 10
        modele = creer_modele_mlp(input_dim, num_classes)
        modele.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1)
        ]
        
        # 4. Entraînement
        print("\n--- Démarrage de l'entraînement ---")
        history = modele.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100, batch_size=32, callbacks=callbacks, verbose=1
        )
        
        # 5. Evaluation & Visualisation
        print("\n--- Evaluation finale sur le jeu de Test ---")
        loss, acc = modele.evaluate(X_test, y_test, verbose=0)
        print(f"Accuracy Test : {acc * 100:.2f}%")
        tracer_courbes(history)
        
        y_pred_prob = modele.predict(X_test)
        y_pred_classes = np.argmax(y_pred_prob, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        afficher_matrice_confusion(y_true_classes, y_pred_classes)
        print("\n--- Rapport de Classification ---")
        print(classification_report(y_true_classes, y_pred_classes, target_names=NOMS_LABELS))

        # 6. Test du système de discrimination (Simulation)
        print("\n--- TEST DU SYSTÈME : PRÉSENTATION D'IMAGES INCONNUES ---")
        indices_aleatoires = random.sample(range(len(X_test)), 5)

        for i in indices_aleatoires:
            vecteur_image = X_test[i]
            index_vrai = np.argmax(y_test[i])
            vrai_label = NOMS_LABELS[index_vrai]
            
            classe_predite, confiance = systeme_discrimination(vecteur_image, modele, NOMS_LABELS)
            resultat = "✅ SUCCÈS" if classe_predite == vrai_label else "❌ ERREUR"
            print(f"Image #{i} (Vrai: {vrai_label}) -> Système: {classe_predite} ({confiance*100:.1f}%) -> {resultat}")

        # 7. Comparaison des hyperparamètres
        comparer_hyperparametres(X_train, y_train, X_val, y_val, X_test, y_test)