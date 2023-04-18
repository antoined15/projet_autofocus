#!/bin/bash

sudo apt update

# Téléchargement de toutes les ressources nécéssaire
echo "Téléchargement de toutes les ressources nécéssaire"

# Création d'un dossier temporaire contenant les ressources
if [ -d "./tmp" ]; then
    echo "Le dossier existe"
else
    mkdir ./tmp
fi
cd ./tmp

# Téléchargement du dépot github
git clone git@github.com:antoined15/projet_autofocus.git autofocus_git

# Téléchargement de Arduino IDE
wget https://downloads.arduino.cc/arduino-1.8.19-linuxaarch64.tar.xz
tar -xf arduino-1.8.19-linuxaarch64.tar.xz

# Téléchargement des drivers Arducam
wget -O install_pivariety_pkgs.sh https://github.com/ArduCAM/Arducam-Pivariety-V4L2-Driver/releases/download/install_script/install_pivariety_pkgs.sh

# Installation de arduino IDE si il n'existe pas
arduino_executable=$(which arduino || which arduino-cli)

if [ -z "$arduino_executable" ]; then
    echo "Arduino n'est pas installé sur le système."
else
    echo "Arduino est installé sur ce système."
    cd arduino-1.8.19/
    sudo ./install.sh
    cd ..
fi

# Installation de l'environnement Arduino
cd arduino-1.8.19/
sudo ./install.sh
cd ..

# Copie des fichier pour arduino IDE
if [ -d "/home/pi/Arduino" ]; then
    echo "Le dossier Arduino existe"
else
    mkdir /home/pi/Arduino
fi

cp -R autofocus_git/test_communication_tourelle/Arduino /home/pi/Arduino

# Installation des drivers caméra
chmod +x install_pivariety_pkgs.sh

./install_pivariety_pkgs.sh -p libcamera_dev
./install_pivariety_pkgs.sh -p libcamera_apps
./install_pivariety_pkgs.sh -p imx519_kernel_driver

# Suppression du dossier tmp
cd ..
cp -R tmp/autofocus_git autofocus_git
rm -rf tmp/

