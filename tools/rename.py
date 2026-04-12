import os
import shutil

def renommer_et_deplacer():
    # --- CHEMINS ---
    script_dir    = os.path.dirname(os.path.abspath(__file__))
    dossier_input = os.path.join(script_dir, "NeedRename")
    dossier_output = os.path.abspath(os.path.join(script_dir, '..', 'data_source', 'data'))

    # Vérification du dossier source
    if not os.path.exists(dossier_input):
        print(f"❌ Erreur : Le dossier '{dossier_input}' n'a pas été trouvé.")
        print("Veuillez créer un dossier nommé 'NeedRename' à côté de ce script.")
        return

    # Création du dossier destination si inexistant
    if not os.path.exists(dossier_output):
        os.makedirs(dossier_output)
        print(f"📁 Dossier de destination créé : {dossier_output}")

    # --- PARAMÈTRES ---
    try:
        hdv_num     = input("Entrez le numéro HDV : ")
        dernier_num = int(input("Entrez le numéro du dernier fichier : "))
    except ValueError:
        print("❌ Erreur : Veuillez entrer des nombres valides.")
        return

    # --- RÉCUPÉRATION DES FICHIERS ---
    fichiers = os.listdir(dossier_input)
    fichiers = [f for f in fichiers if os.path.isfile(os.path.join(dossier_input, f)) and not f.startswith('.')]
    fichiers.sort()

    if not fichiers:
        print("❌ Aucun fichier trouvé dans le dossier NeedRename.")
        return

    print(f"\n📂 {len(fichiers)} fichiers trouvés...")
    print(f"📤 Destination : {dossier_output}\n")

    compteur = dernier_num + 1

    for ancien_nom in fichiers:
        nom_base, extension = os.path.splitext(ancien_nom)

        nouveau_nom    = f"hdv{hdv_num}_{compteur}{extension}"
        ancien_chemin  = os.path.join(dossier_input, ancien_nom)
        nouveau_chemin = os.path.join(dossier_output, nouveau_nom)

        shutil.move(ancien_chemin, nouveau_chemin)

        print(f"✅ {ancien_nom} -> {nouveau_nom}  (déplacé)")

        compteur += 1

    print(f"\n🎉 Terminé ! {compteur - dernier_num - 1} fichiers renommés et déplacés vers 'data_source/data'.")

if __name__ == "__main__":
    renommer_et_deplacer()
    input("\nAppuyez sur Entrée pour quitter...")