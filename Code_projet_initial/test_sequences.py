
import numpy as np
import random

Mire_reel =    [[1, 3, 3, 2, 1, 3, 1, 2, 3, 3, 1, 3, 2, 2, 1], #si = 1, trait. Si = 2, ronds, si 3 = cercle
                [3, 2, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 3, 3, 1], 
                [2, 1, 2, 2, 1, 2, 2, 2, 3, 3, 3, 2 ,3, 3, 1], 
                [1, 3, 1, 3, 3, 3, 1, 3, 2, 2, 2, 3, 1, 2, 3], 
                [1, 1, 3, 1, 3, 1, 2, 3, 2, 1, 1, 3, 1, 2, 3], 
                [2, 1, 3, 3, 2, 1, 2, 1, 2, 1, 3, 2, 2, 1, 2], 
                [2, 3, 1, 1, 2, 3, 3, 1, 3, 3, 1, 2, 1, 1, 2],
                [1, 2, 2, 3, 1, 2, 1, 1, 1, 3, 2, 1, 2, 1, 1], 
                [3, 1, 2, 2, 1, 3, 2, 3, 2, 2 ,3, 1, 2, 3 ,1], 
                [3, 3, 3, 1, 1, 1, 3, 1, 2, 3, 3, 3, 1 ,3 ,2], 
                [1, 3, 3, 2, 2, 1, 1 ,3, 3, 2, 3, 1, 3, 2, 1], 
                [3, 3, 2, 1, 3, 1, 3, 1, 3, 3, 1, 1, 1, 3, 1], 
                [2, 2, 1, 2, 1, 3, 2, 2, 2, 3, 1, 1 ,2, 3, 3], 
                [3, 1, 2, 1, 1, 3, 1, 1, 2, 1, 2, 3, 1, 2, 2], 
                [3, 2, 3, 3, 1, 2, 1, 3, 1, 1, 3, 1, 2, 3, 3]] #correspond à la mire réelle (normalement)

Mire_detectée =[[1, 3, 3, 2, 1, 3, 1, 2, 3, 3, 1, 3, 2, 2, 1], #si = 1, trait. Si = 2, ronds, si 3 = cercle
                [3, 2, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 3, 3, 1], 
                [2, 1, 2, 2, 1, 2, 2, 2, 3, 3, 3, 2 ,3, 3, 1], 
                [1, 3, 1, 3, 3, 3, 1, 3, 2, 2, 2, 3, 1, 2, 3], 
                [1, 1, 3, 1, 3, 1, 2, 3, 2, 1, 1, 3, 1, 2, 3], 
                [2, 1, 3, 3, 2, 1, 2, 1, 2, 1, 3, 2, 2, 1, 2], 
                [2, 3, 1, 1, 2, 3, 3, 1, 3, 3, 1, 2, 1, 1, 2],
                [1, 2, 2, 3, 1, 2, 1, 1, 1, 3, 2, 1, 2, 1, 1], 
                [3, 1, 2, 2, 1, 3, 2, 3, 2, 2 ,3, 1, 2, 3 ,1], 
                [3, 3, 3, 1, 1, 1, 3, 1, 2, 3, 3, 3, 1 ,3 ,2], 
                [1, 3, 3, 2, 2, 1, 1 ,3, 3, 2, 3, 1, 3, 2, 1], 
                [3, 3, 2, 1, 3, 1, 3, 1, 3, 3, 1, 1, 1, 3, 1], 
                [2, 2, 1, 2, 1, 3, 2, 2, 2, 3, 1, 1 ,2, 3, 3], 
                [3, 1, 2, 1, 1, 3, 1, 1, 2, 1, 2, 3, 1, 2, 2], 
                [3, 2, 3, 3, 1, 2, 1, 3, 1, 1, 3, 1, 2, 3, 3]] #correspond à la mire réelle (normalement)


def creation_mire_symb_coord_fictive(mire_detectee):
    mire_symb_coord_fictive = np.zeros((np.size(mire_detectee,0), np.size(mire_detectee,1), 3))

    for i in range(0, np.size(mire_detectee,0)):
        for j in range(0, np.size(mire_detectee,1)):
            mire_symb_coord_fictive[i][j] = [mire_detectee[i][j], random.randint(0, 100), random.randint(0, 100)]

    return mire_symb_coord_fictive


def determination_sequences_mire_reel(mire):
    print("mire", mire)
    posit_secu = []
    for i in range(1, np.size(mire,0) -1):
        for j in range(1, np.size(mire,1) -1):
                #print("position", i, j)
                A = mire[i][j]
                B = mire[i-1][j]
                C = mire[i][j+1]
                D = mire[i+1][j]
                E = mire[i][j-1]
                F = mire[i-1][j-1]
                G = mire[i-1][j+1]
                H = mire[i+1][j+1]
                I = mire[i+1][j-1]
                sequ = [A, B, C, D, E, F, G, H, I]
                posit_secu_uniq = [i, j], sequ
                posit_secu.append(posit_secu_uniq)

    return posit_secu

def determination_sequences_mire_detect(mire):
    posit_secu = []
    for i in range(1, np.size(mire,0) -1):
        for j in range(1, np.size(mire,1) -1):
                A = int(mire[[i][0]][[j][0]][0])
                B = int(mire[[i-1][0]][[j][0]][0])
                C = int(mire[[i][0]][[j+1][0]][0])
                D = int(mire[[i+1][0]][[j][0]][0])
                E = int(mire[[i][0]][[j-1][0]][0])
                F = int(mire[[i-1][0]][[j-1][0]][0])
                G = int(mire[[i-1][0]][[j+1][0]][0])
                H = int(mire[[i+1][0]][[j+1][0]][0])
                I = int(mire[[i+1][0]][[j-1][0]][0])
                sequ = [A, B, C, D, E, F, G, H, I]
                posit_secu_uniq = [int(mire[i][j][1]), int(mire[i][j][2])], sequ  #mettre les bonnes coordonnées
                #print("position secu uniq", posit_secu_uniq)
                posit_secu.append(posit_secu_uniq)

    return posit_secu


def comparaison_sequences(posit_sequence_reel, posit_sequence_detectee, erreur_max):

    corresp_reel_detecte = []
    for pt_reel in posit_sequence_reel:
        for pt_dect in posit_sequence_detectee:
            erreur = 0
            for i in range(len(pt_reel[1])):
                 if pt_reel[1][i] != pt_dect[1][i]:
                     erreur += 1
            if erreur <= erreur_max:
                
                corresp_reel_detecte.append([pt_reel[0], pt_dect[0], pt_reel[1], erreur])
    
    return corresp_reel_detecte


def tri_correspondances(sequences_corresp):

    corresp_reel_detect_trie = sorted(sequences_corresp, key=lambda x: (x[0], x[3]))   #on trie par position puis par erreur
    #on enlève les doublons car on peut avoir plusieurs correspondances pour un point de la mire réelle

    resultat_final = []
    resultat_final.append(corresp_reel_detect_trie[0])
    sub_list = []
    for i in range(1, len(corresp_reel_detect_trie)):

        if corresp_reel_detect_trie[i][0] != corresp_reel_detect_trie[i-1][0]:
            resultat_final.append(corresp_reel_detect_trie[i])
        
    return resultat_final


mire_essai_posit = creation_mire_symb_coord_fictive(Mire_detectée)


#On détermine les séquences de chaque point de la mire réelle et des symboles détectés
posit_sequence_reel = determination_sequences_mire_reel(Mire_reel)
posit_sequence_detectee = determination_sequences_mire_detect(mire_essai_posit)

print(mire_essai_posit)

#On compare les séquences de chaque point de la mire réelle et des symboles détectés
corresp_reeel_detect = comparaison_sequences(posit_sequence_reel, posit_sequence_detectee, erreur_max=3)

#for point in corresp_reeel_detect:
    #print("position réelle", point[0], "\tposition détectée", point[1], "\tséquence", point[2], "\t symbole correspondant", Mire_reel[point[0][0]][point[0][1]], "\t erreur", point[3])

#Certains points de la mire réelle peuvent correspondre à plusieurs points de la mire détectée, il faut donc choisir celui avec le moins d'erreurs
corresp_reeel_detect_trie = tri_correspondances(corresp_reeel_detect)

#for point in corresp_reeel_detect_trie:
    #print("position réelle", point[0], "\tposition détectée", point[1], "\tséquence", point[2], "\t symbole correspondant", Mire_reel[point[0][0]][point[0][1]], "\t nbr erreur", point[3])

#print(corresp_reeel_detect_trie)





matrice_reel_rot_90 = np.rot90(Mire_reel, k=1)

