#include <ax12.h>  
#include <BioloidController.h>

char Controle; 
//Servo du bas ID 1
int Pos_Max_Gauche=800;
int Pos_Max_Droite=200;
//Servo du Haut ID 2
int Pos_Max_Bas=310;
int Pos_Max_Haut=650;

// the setup routine runs once when you press reset:
void setup() {   
  Serial.begin(9600);
  delay(3000);    
  
  //ax12SetRegister(3,3,2);
  //ax12SetRegister(-1,3,1);

  int identifiant1 =ax12GetRegister(1,3,1); 
  Serial.print("le servo 1 possede l'idenifiant suivant : ");
  Serial.println (identifiant1);
  
  int identifiant2 =ax12GetRegister(2,3,1); 
  Serial.print("le servo 2 possede l'idenifiant suivant : ");
  Serial.println (identifiant2);
  
  int identifiant3 =ax12GetRegister(3,3,1); 
  Serial.print("le servo 3 possede l'idenifiant suivant : ");
  Serial.println (identifiant3);
  
  int identifiant4 =ax12GetRegister(4,3,1); 
  Serial.print("le servo 4 possede l'idenifiant suivant : ");
  Serial.println (identifiant4);
  
  int identifiant5 =ax12GetRegister(5,3,1); 
  Serial.print("le servo 5 possede l'idenifiant suivant : ");
  Serial.println (identifiant5);
  
  int identifiant6 =ax12GetRegister(6,3,1); 
  Serial.print("le servo 6 possede l'idenifiant suivant : ");
  Serial.println (identifiant6);

  int Position_Bas=GetPosition(1);
  int Position_Haut=GetPosition(2);
  Serial.print("Position BAS : ");
  Serial.println (Position_Bas);
  Serial.print("Position HAUT : ");
  Serial.println (Position_Haut);  
  delay(2000);
  
  //INIT MODE DE LA TOURELLE
  //Si les valeurs dans les registres (6,7) et (8,9) sont différents de 0, alors la tourelle est en mode joint mode. Les angles sont limités.
  //La vitesse actuelle retournée sera une valeur entre 0 et 1023.
  //ax12SetRegister(2,3,3);
  //ax12SetRegister(2,3,3);

  // Adresse 1 au moteur bas, 2 moteur haut    
  //ax12SetRegister(2,3,3);
  //ax12SetRegister(1,3,2);
  //ax12SetRegister(3,3,1);
}

// the loop routine runs over and over again forever:
void loop() {
    
    if (Serial.available() > 0 ) {
      Controle = Serial.read();
      //Serial.println(Controle);
    }
    if ((Controle == 'r') || (Controle == 'R')){
      dxlSetGoalSpeed(1,0);
      dxlSetGoalSpeed(2,0);
      SetPosition(1,Pos_Max_Gauche);
      SetPosition(2,Pos_Max_Bas);      
      arrivee2(1,Pos_Max_Gauche,2,Pos_Max_Bas);
      
      SetPosition(1,Pos_Max_Droite);
      arrivee1(1,Pos_Max_Droite);
      
      SetPosition(2,Pos_Max_Haut);
      arrivee1(2,Pos_Max_Haut);
      
      SetPosition(1,Pos_Max_Gauche);
      arrivee1(1,Pos_Max_Gauche);
    }
    if ((Controle == 'q') || (Controle == 'Q')){
      dxlSetGoalSpeed(1,200);
      dxlSetGoalSpeed(2,200);
      
      SetPosition(1,Pos_Max_Gauche);
      SetPosition(2,Pos_Max_Bas);
      arrivee2(1,Pos_Max_Gauche,2,Pos_Max_Bas);
      
      SetPosition(1,Pos_Max_Droite);
      arrivee1(1,Pos_Max_Droite);

      SetPosition(2,Pos_Max_Haut);
      arrivee1(2,Pos_Max_Haut);
      
      SetPosition(1,Pos_Max_Gauche);
      arrivee1(1,Pos_Max_Gauche);
    }
    if ((Controle == 'p') || (Controle == 'P')){
      Serial.print("Position Servo 1: ");
      Serial.println(GetPosition(1));
      Serial.print("Position Servo 2: ");
      Serial.println(GetPosition(2));
      Controle = 'a';
    }
}

void arrivee1(int ID, int POS){
  while ((GetPosition(ID)>POS+2)||(GetPosition(ID)<POS-2)){;}
}

void arrivee2(int ID1, int POS1,int ID2, int POS2){
  while ((GetPosition(ID1)>POS1+2)||(GetPosition(ID1)<POS1-2)||(GetPosition(ID2)>POS2+2)||(GetPosition(ID2)<POS2-2)){;}
}
