# Déployer la démo sur Hugging Face Spaces

1. Créer un compte gratuit sur huggingface.co.
2. *New → Space* : nom `moteur-reco`, SDK **Gradio**, matériel **CPU basic (gratuit)**, visibilité **Public**.
3. Créer un jeton d'accès : *Settings → Access Tokens → New token*, rôle **Write**. Le garder pour soi.
4. Assembler le dossier : `python deploy/build_space.py` (crée `space_build/`).
5. Pousser :
   ```bash
   cd space_build
   git init -b main
   git add .
   git commit -m "Demo"
   git remote add space https://huggingface.co/spaces/<ton-nom>/moteur-reco
   git push space main
   ```
   Identifiant : ton nom d'utilisateur ; mot de passe : le jeton (pas ton mot de passe de compte).
6. Le Space construit l'image (5 à 10 min la première fois), puis l'app s'entraîne au démarrage (~1 min).
7. Le lien final : `https://huggingface.co/spaces/<ton-nom>/moteur-reco`.

Les Spaces gratuits s'endorment après inactivité : le premier visiteur attend le redémarrage.
