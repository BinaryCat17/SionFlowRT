#include <stdio.h>
#include <math.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#define PARALLEL _Pragma("omp parallel for")
#else
#define PARALLEL
#endif

// Буферы тензоров
{% for node in nodes -%}
{{ node.c_type }} buffer_{{ node.id }}[{{ node.size }}];
{% endfor %}

void execute() {
{% for group in groups %}
    /* --- Fusion Group (Shape: {{ group.shape }}) --- */
{{ group.loops_open -}}
{%- for op in group.operations %}
    {{ group.indent }}// {{ op.id }}
    {{ group.indent }}{{ op.body }}
{% endfor -%}
{{ group.loops_close }}
{%- endfor %}
}

int main() {
    execute();
    printf("Calculation finished. Result[0]: %f\n", (double)buffer_out[0]);
    return 0;
}