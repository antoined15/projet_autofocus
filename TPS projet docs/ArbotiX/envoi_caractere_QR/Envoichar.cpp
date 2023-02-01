// Includes
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include "wiringSerial.h"


#include <iostream>
#include <string>
using namespace std;

 
 //Compile with : gcc -o test Envoichar.cpp -lwiringPi -lstdc++
 // Execute with ./test
 
 int main(){
	 
	 int fd;
	int baudrate = 9600;
	char c,c_read;
	
	 if( (fd = serialOpen("/dev/ttyUSB0",baudrate)) < 0){ //vérifier le port série avec "dmesg" dans le terminal
		 fprintf(stderr, "Impossible d'établir la com. série : %s\n", strerror(errno));
		return 1; 
	 }
		 
	while(1){
		
			//Envoi d'une instruction
			printf("\nInstructions à envoyer Q/R/P/autres (= arret): ");
			scanf(" %c",&c); 
			serialPutchar(fd, c);
			//serialFlush(fd);
			
			if(c == 'p')


			
					
				while(serialDataAvail(fd)){
					c_read = serialGetchar(fd);
					printf("%c", c_read);
				}
		
			
	}
	 
	 
	 
 }

//dmesg
