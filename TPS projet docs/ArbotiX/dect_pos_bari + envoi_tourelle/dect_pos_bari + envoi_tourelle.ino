#include <ax12.h>

// TRAME DE POSITION AU FORMAT
// S12345678. pour servo du bas : 1234 pour ID1 et 5678 pour ID2

String trame ="";
String xpos, ypos = "";
int xposi,yposi = 0;

int Pos_Max_Gauche=800;
int Pos_Max_Droite=200;
int Pos_Max_Bas=310;
int Pos_Max_Haut=650;
char controle = 'F'; //E = erreur / F = fonctionnement

// the setup routine runs once when you press reset:
void setup() {   
  Serial.begin(57600);
  Serial.setTimeout(1000); // ms = timeout, à ajuster.
  //Initial position
  SetPosition(1,500);
  SetPosition(2,460);  
  trame ="";    
}

// the loop routine runs over and over again forever:
void loop() {
  
   if(trame.length() == 8) {// on attend la totalité de la trame

       //Reinit de controle :
       controle = 'F';
       
      //Extration des valeurs de positions dans la trame
      xpos = trame.substring(0,4);
      ypos = trame.substring(4,8);

      //Cast en entier
      xposi = xpos.toInt();
      yposi = ypos.toInt();

      //Test des positions dans les limites : on repère une erreur de conversion
      if(xposi > Pos_Max_Gauche){
        xposi = Pos_Max_Gauche;
        controle = 'E';
      }
      
      if(xposi < Pos_Max_Droite){
        xposi = Pos_Max_Droite;
        controle = 'E';
      }
      
      if(yposi > Pos_Max_Haut) {
        yposi = Pos_Max_Haut;
        controle = 'E';
      }
      
      if(yposi < Pos_Max_Bas) {
        yposi = Pos_Max_Bas;
        controle = 'E';
      }
      
      //Ecriture des positions si pas d'erreur

      if(controle == 'F'){
      SetPosition(1,xposi);
      SetPosition(2,yposi);

      //Attente arrivée
      //arrivee2(1, xposi,2, yposi);

      //Acquittement
      //Serial.write('F');
      //Serial.flush();
      }
      else {
      //On signal l'erreur au raspi
      //Serial.write('E');
      //Serial.flush();
      }
      
   //On réinit la trame : serialEvent pourra reprendre la main
   trame = ""; 
   }
}

void serialEvent(){
  
 String poubelle = "";
 
 while(trame.length() != 8){
    if(Serial.available()){
      poubelle = Serial.readStringUntil('S'); //On lit jusqu'à un début de chaine et on supprime du buffer
      trame = Serial.readStringUntil('.'); // On lit la trame jusqu'à un .
    }   
 }
  //Serial.print(trame);

  
}



void arrivee2(int ID1, int POS1,int ID2, int POS2){
  while (((GetPosition(ID1)>POS1+2)||(GetPosition(ID1)<POS1-2))&&((GetPosition(ID2)>POS2+2)||(GetPosition(ID2)<POS2-2))){;}
}
