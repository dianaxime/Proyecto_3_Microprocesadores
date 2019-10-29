/*
 
 Maria Ines Vasquez 18250
 Camila Gonzalez 18398
 Diana de Leon 18---
 Proyecto 3 de Microprocesadores
 
*/
//#include <iostream>
//#include <fstream>
//constaantes necesarias para inicializar sensor
int rainPin = A0;
int greenLED = 6;
int redLED = 7;
// you can adjust the threshold value
int thresholdValue = 200;
String fileName;

//setup para guardar lecturas en el documento .csv
void setup(){
  pinMode(rainPin, INPUT);
  pinMode(greenLED, OUTPUT);
  pinMode(redLED, OUTPUT);
  digitalWrite(greenLED, LOW);
  digitalWrite(redLED, LOW);
  Serial.begin(9600);
  Serial.println("CLEAR DATA");
  Serial.println("LABEL,Lecturas sensor");
  Serial.println("RESETTIMER");
}

void loop() {
  // read the input on analog pin 0:
  int sensorValue = analogRead(rainPin); //valor de la lectura de humedad
  //proceso para guarda en el .csv
  Serial.print("DATA,TIME,TIMER,");
  Serial.println(sensorValue);
  
  delay(500);
}
