#include <ax12.h>

// TRAME DE POSITION AU FORMAT
// S12345678. pour servo du bas : 1234 pour ID1 et 5678 pour ID2

String trame ="";
int xpos, ypos = 0;

int max_left_pos=800;
int max_right_pos=200;
int max_down_pos=310;
int max_up_pos=650;

char control = 'O'; //E = error / O = operating

// the setup routine runs once when you press reset:
void setup() {   
  Serial.begin(57600);
  Serial.setTimeout(1000); // ms = timeout, to adapt.

  //Initial speed
  dxlSetGoalSpeed(1,0);
  dxlSetGoalSpeed(2,0);

  //Initial position
  SetPosition(1,500);
  SetPosition(2,460);  

  trame ="";    
}

// the loop routine runs over and over again forever:
void loop() {
  
   if(trame.length() == 8) {// on attend la totalité de la trame

      //Reinit de controle :
      control = 'O';
       
      //Extration des valeurs de positions dans la trame
      xpos = trame.substring(0,4).toInt();
      ypos = trame.substring(4,8).toInt();


      //Check if each position is valid
      if(xpos > max_left_pos){
        xpos = max_left_pos;
        control = 'E';
      }
      if(xpos < max_right_pos){
        xpos = max_right_pos;
        control = 'E';
      }
      if(ypos > max_up_pos) {
        ypos = max_up_pos;
        control = 'E';
      }
      if(ypos < max_down_pos) {
        ypos = max_down_pos;
        control = 'E';
      }
      
      //Ecriture des positions si pas d'erreur

      if(control == 'O'){
      SetPosition(1,xpos);
      SetPosition(2,ypos);

      //Attente arrivée
    //  arrivee2(1, xposi,2, yposi);

      //Acquittement
     // Serial.write('F');
     // Serial.flush();
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
  
 String trash = "";
 
 while(trame.length() != 8){
    if(Serial.available()){
      trash = Serial.readStringUntil('S'); //On lit jusqu'à un début de chaine et on supprime du buffer
      trame = Serial.readStringUntil('.'); // On lit la trame jusqu'à un .
    }   
 }
  //Serial.print(trame);

  
}



void arrivee2(int ID1, int POS1,int ID2, int POS2){
  while (((GetPosition(ID1)>POS1+2)||(GetPosition(ID1)<POS1-2))&&((GetPosition(ID2)>POS2+2)||(GetPosition(ID2)<POS2-2))){;}
}
