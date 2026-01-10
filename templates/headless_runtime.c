#include <stdio.h>
#include <stdint.h>
#include <string.h>

/* --- Parameters --- */
{% for name, value in parameters -%}
#define {{ name }} {{ value }}
{% endfor %}

extern void execute();

int main(int argc, char* argv[]) {
    printf("Running in Headless Mode...\n");
    
    // Выполняем 10 итераций для примера
    for(int i = 0; i < 10; ++i) {
        printf("Iteration %d...\n", i);
        execute();
    }
    
    printf("Headless execution finished.\n");
    return 0;
}

