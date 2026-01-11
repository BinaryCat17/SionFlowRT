#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>

/* --- Parameters --- */
{% for name, value in parameters -%}
#define {{ name }} {{ value }}
{% endfor %}

extern void execute_all();

{% for id, res in orchestration.resources -%}
extern float* resource_{{ id }};
{% endfor %}

int main(int argc, char* argv[]) {
    printf("SionFlow Headless Runtime Starting...\n");
    
    uint32_t frames = 100; // Run for 100 frames
    for(uint32_t f = 0; f < frames; ++f) {
        // Here we could inject mock data into resources
        execute_all();
    }

    printf("Done.\n");
    return 0;
}