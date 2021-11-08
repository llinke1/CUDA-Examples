

#include <iostream>
#include <chrono>

int main()
{
    // Declare and define arrays
    const int arraySize = 10240; // Number of elements in array
    int a[arraySize]; //Summand 1
    int b[arraySize]; // Summand 2
    int c[arraySize]; // Result array

    // Fill arrays
    for (int i = 0; i < arraySize; i++)
    {
        a[i] = i;
        b[i] = 2 * i;
    };


    // Start time measurement
    auto start = std::chrono::high_resolution_clock::now();

   // Do calculation
    for (int i = 0; i < arraySize; i++)
    {
        c[i] = a[i] * a[i];// +b[i] * b[i];
    };

    // Stop time measurement
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = stop - start;

   
    // Output
  /*  printf("Result is \n");
    for (int i = 0; i < arraySize; i++)
    {
        printf("%d \n", c[i]);
    };*/

    printf("Needed %f seconds for adding on CPU \n", elapsed.count());

 
    return 0;

}

