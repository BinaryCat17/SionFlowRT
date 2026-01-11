#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#define PARALLEL _Pragma("omp parallel for")
#else
#define PARALLEL
#endif

/* --- Parameters --- */
{% for name, value in parameters -%}
#define {{ name }} {{ value }}
{% endfor %}

/* --- Tensor Buffers --- */
{% for prog in programs %}
/* Program: {{ prog.id }} */
{% for node in prog.nodes -%}
float buffer_{{ prog.id }}_{{ node.id }}[{{ node.size }}];
{% endfor %}
{% for out in prog.outputs -%}
#define buffer_{{ prog.id }}_{{ out.alias }} buffer_{{ prog.id }}_{{ out.real_id }}
{% endfor %}
{% endfor %}

/* --- Execution Functions --- */
{% for prog in programs %}
void execute_{{ prog.id }}() {
    {% for node in prog.nodes %}
    {% if not node.is_input %}
    // Node: {{ node.id }}
    PARALLEL
    for(int i = 0; i < {{ node.size }}; ++i) {
        {{ node.body }}
    }
    {% endif %}
    {% endfor %}
}
{% endfor %}

void execute_all() {
    {% for prog in programs -%}
    execute_{{ prog.id }}();
    {% endfor %}
}