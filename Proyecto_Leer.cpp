/**
 * 
 * PROYECTO NO. 3 DE MICROPROCESADORES
 * 22 DE OCTUBRE DE 2019
 * 
 *
 * Maria Ines Vasquez, 18250
 * Paula Camila Gonzalez, 18398
 * Diana Ximena de Leon, 18607
 *
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdlib.h>
#include <vector>

using namespace std;

int main(){
  ifstream myFile("datos.csv");
  if(!myFile.is_open()){
    cout<<"File failed open"<<endl;
    return 0;
  }
  vector<double> tiempos;
  vector<double> mediciones;
  double times, values;
  string hora, medicion;
  string line;

while(getline(myFile,line)){
  stringstream ss(line);
  getline(ss, hora, ',');
  times = stod(hora);
  tiempos.push_back(times);
  getline(ss, medicion, ',');
  values = stod(medicion);
  mediciones.push_back(values);
  cout<< "Hora: "<< times << "\n";
  cout<< "Valor: "<< values << "\n";
}
  cout << "Cantidad de datos: " << tiempos.size() << '\n';
  myFile.close();
}