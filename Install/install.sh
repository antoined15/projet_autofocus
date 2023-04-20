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
cd tmp/

# Téléchargement du dépot github
#git clone git@github.com:antoined15/projet_autofocus.git autofocus_git
wget -O autofocus.zip https://github.com/antoined15/projet_autofocus/archive/refs/heads/main.zip
unzip autofocus.zip 
mv projet_autofocus-main autofocus_git

# Téléchargement de Arduino IDE
wget https://downloads.arduino.cc/arduino-1.8.19-linuxaarch64.tar.xz
tar -xf arduino-1.8.19-linuxaarch64.tar.xz

# Téléchargement des drivers Arducam
wget -O install_pivariety_pkgs.sh https://github.com/ArduCAM/Arducam-Pivariety-V4L2-Driver/releases/download/install_script/install_pivariety_pkgs.sh

# Installation de arduino IDE si il n'existe pas
arduino_executable=$(which arduino || which arduino-cli)

if [ -z "$arduino_executable" ]; then
    echo "Arduino n'est pas installé sur le système."
    cd arduino-1.8.19/
    sudo ./install.sh
    cd ..
else
    echo "Arduino est installé sur ce système."
fi

# Copie des fichier pour arduino IDE
if [ -d "/home/pi/Arduino" ]; then
    echo "Le dossier Arduino existe"
else
    mkdir /home/pi/Arduino
fi

cp -R autofocus_git/Test/Tourelle/Arduino /home/pi/

# Installation des drivers caméra
chmod +x install_pivariety_pkgs.sh

sudo apt update

./install_pivariety_pkgs.sh -p libcamera_dev
./install_pivariety_pkgs.sh -p libcamera_apps
./install_pivariety_pkgs.sh -p imx519_kernel_driver
./install_pivariety_pkgs.sh -p 64mp_pi_hawk_eye_kernel_driver

# Installation des libraries python
pip install numpy==1.24.2 matplotlib==3.7.1 mplcursors==0.5.2 opencv-python-headless==4.7.0.72 pyqt5==5.15.2 pyserial==3.5b0 picamera2==0.3.6 argparse==1.4.0

# Pour eviter des problèmes (pas sur)
pip uninstall opencv-python-headless -y
pip install opencv-python --upgrade
pip uninstall opencv-python
pip install opencv-python-headless

# Suppression du dossier tmp
cd ..
cp -R tmp/autofocus_git autofocus_git
rm -rf tmp/

