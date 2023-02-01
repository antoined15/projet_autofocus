/* Amir MERIMECHE - Alexis FRIEDRICH
 * FIP EII 2A
 * Christophe DOIGNON
 * 
 * Create, compile and execute :
 * 
 * mkdir build  
#include <iostream>
#include <raspicam/raspicam.h>
#include <unistd.h> //for usleep()

#include <string.h>
#include "wiringSerial.h"   

//Variables globales
int fd;

using namespace std;

#define WIDTH 640 //Allowable width : 320 / 640 / 1280
#define HEIGHT 480 //Allowable height : 240 / 480 / 960



//Functions
int envoi_pos(int x, int y);


int main ( int argc,char **argv ) {
	
    raspicam::RaspiCam Camera; //Camera object
    


    //Picture format
    // Allowable values :  RASPICAM_FORMAT_GRAY / RASPICAM_FORMAT_RGB / RASPICAM_FORMAT_BGR / RASPICAM_FORMAT_YUV420
    Camera.setFormat(raspicam::RASPICAM_FORMAT_GRAY);
    
    //Capture size
    //Allowable width : 320 / 640 / 1280
    //Allowable height : 240 / 480 / 960
    Camera.setCaptureSize(WIDTH,HEIGHT);
    
    //Rotation in our project -90°
    Camera.setRotation(-90);
    
    //Vertical and Horizontal flip
    Camera.setHorizontalFlip(1);
    Camera.setVerticalFlip(1);
    
    //open the camera
    cout<<"Opening Camera..."<<endl;
    if ( !Camera.open()) {cerr<<"Error opening camera"<<endl;return -1;}
    //Wait until camera stabilizes
    cout<<"Sleeping for 3 secs"<<endl;
    usleep(3000000);
    
 
    
   
    //memory allocation for camera buffer
    unsigned long bytes = Camera.getImageBufferSize();
    unsigned int width = Camera.getWidth();
    unsigned int height = Camera.getHeight();


    unsigned char *data=new unsigned char[bytes];
    

      //Liaison série
	int baudrate = 57600;    
		 if( (fd = serialOpen("/dev/ttyUSB0",baudrate)) < 0){ //vérifier le port série avec "dmesg" dans le terminal
		 fprintf(stderr, "Impossible d'établir la com. série en USB0, essai en USB1: %s\n", strerror(errno));
		 
		 if( (fd = serialOpen("/dev/ttyUSB1",baudrate)) < 0){ //vérifier le port série avec "dmesg" dans le terminal
			fprintf(stderr, "Impossible d'établir la com. série en USB1, essai en USB2: %s\n", strerror(errno));
			
			 if( (fd = serialOpen("/dev/ttyUSB2",baudrate)) < 0){ //vérifier le port série avec "dmesg" dans le terminal
				fprintf(stderr, "Impossible d'établir la com. série en USB2, echec: %s\n", strerror(errno));
				return 1; 
				}
			}
	 
	 }
    
    
    //Loop : searching for High intensity pixels (higher than "seuil") 
    int seuil = 205;
    int i = 0;
    int ligne, colonne = 0;
    
    //fenêtre contenant la tâche
    int xmoy,ymoy=0;
    int xconsigne = 500;
    int yconsigne = 500;
    int pas = 5; // pas de consigne
    int pasMax = 30;
    int pasMin = 5;
    int tol = 50; // tolérance en pixel (fenetre autour du centre)
    int nb_pixel = 0;
    int sum_x = 0;
    int sum_y = 0;
    
    Camera.startCapture();
    
    while(1){
       
        //Valeurs initiales
         sum_x = 0;
         sum_y = 0;
         nb_pixel = 0;

    //capture
    Camera.grab();
    //extraction de l'image au format RGB
    Camera.retrieve ( data,raspicam::RASPICAM_FORMAT_IGNORE);//get camera image RASPICAM_FORMAT_IGNORE => use the format defined upper
    
    
    
    //Analyse
        for(i = 0; i < (width*height); i++){
            
            
            if(data[i] > seuil){ 
                data[i] = 255;
                ligne = i/width;
                colonne = i - ligne*width;
                
                //Sum
                sum_x = sum_x + colonne;
                sum_y = sum_y + ligne;
                nb_pixel = nb_pixel +1;
                
            }
            else
                data[i] = 0;
        
        }
        
        //Baricentre
        if(nb_pixel > 0){
            xmoy = sum_x /nb_pixel;
            ymoy = sum_y /nb_pixel;
            
            
            //printf("\ncentre de la tâche : (x = %d ; y = %d)\n",xmoy,ymoy);

            if(xmoy > (width/2 + tol)){
                pas = (pasMax-pasMin)*(xmoy-width/2)/(width/2) + 5;
                xconsigne = xconsigne - pas;
            }
            else if(xmoy < (width/2 - tol)){
                pas = (pasMax-pasMin)*(width/2-xmoy)/(width/2) + 5;
                xconsigne = xconsigne + pas;
            }
            
            if(ymoy > (height/2 + tol)){
                pas = (pasMax-pasMin)*(ymoy-height/2)/(height/2) + 5;
                yconsigne = yconsigne - pas;
            }
            else if(ymoy < (height/2 - tol)){
                pas = (pasMax-pasMin)*(height/2-ymoy)/(height/2) + 5;
                yconsigne = yconsigne + pas;
            }
        
            //envoi à la tourelle
            envoi_pos(xconsigne,yconsigne);
        } 
        

      
    }
 
    
    //libération de la mémoire  
    delete data;
    Camera.release();
    close(fd);	
    return 0;
}


int envoi_pos(int x, int y) {
    
    //Chaine à transmettre
    char pos[11]="";
    
    //Valeurs
    char xpos[5]="";
    char ypos[5]="";
    char x4digits[4]="";
    char y4digits[4]="";
    
    //Verifier les limites
    if(x > 800) x =800;
    if(y > 650) y = 650;
    if(x < 200) x = 200;
    if(y < 310) y = 310;
    
    //Ajout des 4 caractères de positions x
    if(x < 10) {
        snprintf(x4digits, 2, "%d", x);
        strcat(xpos,"000");
        strcat(xpos,x4digits);
    }
    else if(x < 100) {
        snprintf(x4digits, 3, "%d", x);
        strcat(xpos,"00");
        strcat(xpos,x4digits);
    }
    else if(x < 1000) {
        snprintf(x4digits, 4, "%d", x);
        strcat(xpos,"0");
        strcat(xpos,x4digits);
    }
    else {
        snprintf(x4digits, 5, "%d", x);
        strcat(xpos,x4digits);
    }
    
     //Ajout des 4 caractères de positions y
    if(y < 10) {
        snprintf(y4digits, 2, "%d", y);
        strcat(ypos,"000");
        strcat(ypos,y4digits);
    }
    else if(y < 100) {
        snprintf(y4digits, 3, "%d", y);
        strcat(ypos,"00");
        strcat(ypos,y4digits);
    }
    else if(y < 1000) {
        snprintf(y4digits, 4, "%d", y);
        strcat(ypos,"0");
        strcat(ypos,y4digits);
    }
    else {
        snprintf(y4digits, 5, "%d", y);
        strcat(ypos,y4digits);
    }
    

    //Construction de la trame à envoyer
    strcat(pos,"S");
    strcat(pos,xpos);
    strcat(pos,ypos);
    strcat(pos,".");
    
    serialFlush(fd); //vide les buffers tx et rx
    serialPuts(fd,pos); //envoi
    
    usleep(3*1000); // total en ns
    
    
       
		
     return 0;
    
}
