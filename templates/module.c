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
{% for op in operations %}
    // Node: {{ op.id }} (Shape: {{ op.shape }})
    {% if op.is_input -%}
    for(int i = 0; i < {{ op.size }}; ++i) buffer_{{ op.id }}[i] = 1.0f;
    {%- else -%}
    {{ op.loops_open }}
    {{ op.indent }}    {{ op.body }}
    {{ op.loops_close }}
    {%- endif %}
{% endfor %}
}

int main() {
    execute();
    printf("Calculation finished. Result[0]: %f\n", (double)buffer_out[0]);
    return 0;
}

