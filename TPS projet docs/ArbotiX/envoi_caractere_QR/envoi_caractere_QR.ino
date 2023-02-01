#include <ax12.h>  
#include <BioloidController.h>

char Controle; 
//Servo du bas ID 1
int Pos_Max_Gauche=800;
int Pos_Max_Droite=200;
//Servo du Haut ID 2
int Pos_Max_Bas=295;
int Pos_Max_Haut=650;

// the setup routine runs once when you press reset:
void setup() {   
  Serial.begin(9600);
  delay(2000);    

  int identifiant1 =ax12GetRegister(1,3,1); 
  Serial.print("le servo 1 possede l'idenifiant suivant : ");
  Serial.println (identifiant1);
  
  int identifiant2 =ax12GetRegister(2,3,1); 
  Serial.print("le servo 2 possede l'idenifiant suivant : ");
  Serial.println (identifiant2);

  int Position_Bas=GetPosition(1);
  int Position_Haut=GetPosition(2);
  Serial.print("Position BAS : ");
  Serial.println (Position_Bas);
  Serial.print("Position HAUT : ");
  Serial.println (Position_Haut);  
  delay(2000);

// Adresse 1 au moteur bas, 2 moteur haut    
  //ax12SetRegister(2,3,3);
  //ax12SetRegister(1,3,2);
  //ax12SetRegister(3,3,1);
}

// the loop routine runs over and over again forever:
void loop() {
    
    if (Serial.available() == 1) {
      Controle = Serial.read();
    }
    if ((Controle == 'r') || (Controle == 'R')){
      dxlSetGoalSpeed(1,0);
      dxlSetGoalSpeed(2,0);
      SetPosition(1,Pos_Max_Gauche);
      SetPosition(2,Pos_Max_Bas);      
      delay(500);
      SetPosition(1,Pos_Max_Droite);
      delay(500);
      //int Vit1=GetGoalSpeed(1);
      //Serial.println (Vit1);
      //int Vit2=GetGoalSpeed(2);
      //Serial.println (Vit2);
      SetPosition(2,Pos_Max_Haut);
      delay(500);
      SetPosition(1,Pos_Max_Gauche);
      delay(500);
    }
    if ((Controle == 'q') || (Controle == 'Q')){
      dxlSetGoalSpeed(1,200);
      dxlSetGoalSpeed(2,200);
      SetPosition(1,Pos_Max_Gauche);
      SetPosition(2,Pos_Max_Bas);
      delay(1500);
      SetPosition(1,Pos_Max_Droite);  
      delay(1500);
      SetPosition(2,Pos_Max_Haut);
      delay(1500);
      SetPosition(1,Pos_Max_Gauche);
      delay(1500);
    }
      

}
