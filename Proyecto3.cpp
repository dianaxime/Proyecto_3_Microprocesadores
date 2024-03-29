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


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include<string.h>
#include <sstream>

// Libreria para la ejecucion de la subrutinas en CUDA
#include <cuda_runtime.h>

using namespace std; 

// CUDA Kernel
__global__ void
Calculardesv(const float *A, float *C, int numElements, float promedio)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        C[i] = pow ((A[i] - promedio), 2);
    }
}

int main(int argc, char *argv[])
{
    // Lectura de datos de archivo .CSV
    ifstream lectura;
    lectura.open("datos.csv",ios::in);

    // Revision de los valores de retorno de las llamadas a CUDA
    cudaError_t err = cudaSuccess;


    // Inicializamos el valor del tamaño en 75,000
    int numElements = 75004;

    size_t size = numElements * sizeof(float);

    // Asignar el valor de entrada del vector de mediciones
    float *h_hum = (float *)malloc(size);

    // Asignar el valor de entrada del vector del tiempo de las mediciones
    float *h_tempv = (float *)malloc(size);

    // Asignar el valor de entrada del vector de la desviacion 
    float *h_desv = (float *)malloc(size);

    // Verificacion de las asignaciones
    if (h_hum == NULL || h_desv == NULL || h_tempv == NULL)
    {
        fprintf(stderr, "ERROR al asignar los vectores del host\n");
        exit(EXIT_FAILURE);
    }

    float h_temp = 0;

    // Introducir los valores del medidos por el sensor al arreglo
    string linea, dato;

    // Lee la linea del archivo
    int fila, col;
    for (fila=0;  getline(lectura, linea); fila++ )
    {
        stringstream registro(linea);

        // Para cada linea separa los datos (tiempo y humedad)
        for (col=0;  getline(registro, dato, ';'); col++ )
        {
            // Dependiendo de cual es la columna del dato lo guarda en ese vector 
            switch(col){
                case 0:
                    h_tempv[fila] = strtof((dato).c_str(),0);
                break; 
                case 1:
                    h_hum[fila] = strtof((dato).c_str(),0);
                break;
            }
        }
        h_desv[fila] = 0;
    }

    numElements = fila;

    // Asignar el vector de entrada del device de mediciones
    float *d_hum = NULL;
    err = cudaMalloc((void **)&d_hum, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "ERROR al asignar el vector DEVICE con las mediciones (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Asignar el vector de entrada del device de la desviacion
    float *d_desv = NULL;
    err = cudaMalloc((void **)&d_desv, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "ERROR al asignar el vector DEVICE con la desviacion (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copiar el vector de la memoria del host a la memoria del device
    err = cudaMemcpy(d_hum, h_hum, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "ERROR al copiar el vector de mediciones del host al device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float suma, promedio;
    // Calculo del promedio
    for (int i = 0; i < numElements; ++i)
    {
        suma += h_hum[i];
    }

    promedio = suma/numElements;

    // Lanzamiento del CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel lanzado con %d bloques de %d hilos\n", blocksPerGrid, threadsPerBlock);
    Calculardesv<<<blocksPerGrid, threadsPerBlock>>>(d_hum, d_desv, numElements, promedio);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Fallo en lanzamiento del Kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copiar el resultado de la memoria del device a la memoria del host 
    err = cudaMemcpy(h_desv, d_desv, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "ERROR en la copia de datos del device al host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float desv, sumatoria;
    // Terminar el calculo de la desviacion estandar
    for (int i = 0; i < numElements; ++i)
    {
        sumatoria += h_desv[i];
    }

    desv = pow ((sumatoria/numElements), 0.5);

    // Mostrar el resultado en pantalla
    printf("El promedio de humedad es de %f con una desviacion estandar de %f \n", promedio, desv);

    int i=0; 

    //Observar cuanto tiempo se tardo en secarse la planta

    while (i < numElements && h_hum[i]<340)
    {
        h_temp = h_tempv[i] / 3600;
        i++;
    }
    
    printf("Se tardo %f horas en secarse la planta\n", h_temp);

    // Liberar la memoria del device
    err = cudaFree(d_hum);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "ERROR al liberar la memoria del device del vector de mediciones (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_desv);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "ERROR al liberar la memoria del device del vector de la desviacion (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Liberar la memoria del host
    free(h_hum);
    free(h_desv);
    free(h_tempv);

    // Reiniciar el Device 
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "ERROR al reiniciar el device error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return 0;
}