import sys
import os

def switch_camera(camera):
    if camera not in ['16mp','64mp']:
        print("Erreur: veuillez entrer '16mp' ou '64mp' comme argument.")
        return False
    
    with open('/boot/config.txt', 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if 'dtoverlay=arducam_64mp' in line:
            if camera == '64mp':
                new_lines.append(line.replace('#',''))
            else:
                new_lines.append('#' + line.lstrip('#'))
        elif 'dtoverlay=imx519' in line:
            if camera == '16mp':
                new_lines.append(line.replace('#',''))
            else:
                new_lines.append('#' + line.lstrip('#'))
        else:
            new_lines.append(line)
    
    with open('/boot/config.txt','w') as f:
        f.writelines(new_lines)

    return True

def user_confirmation():
    while True:
        user_input = input("Voulez-vous redemarrer maintenant? (O/N): ").lower()
        if user_input in ['o','oui','y','yes']:
            return True
        elif user_input in ['n','non','no','n']:
            return False
        else:
            print("Entree invalide. Repondez par O(oui) ou N(non).")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage : sudo python switch_camera.py [64mp|16mp]")
    else:
        if (switch_camera(sys.argv[1])):
            print("Le driver de la camera a ete change.")
            if user_confirmation():
                print("Redemarrage en cours...")
                os.system('sudo reboot')
            else:
                print("Redemarrez manuellement pour appliquer les modifications.")