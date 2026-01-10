#include <stdio.h>
#include <math.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#define PARALLEL _Pragma("omp parallel for")
#else
#define PARALLEL
#endif

/* --- Tensor Buffers --- */
{% for node in nodes -%}
#define buffer_{{ node.id }}_SIZE {{ node.size }}
{% if node.init_values -%}
{{ node.c_type }} buffer_{{ node.id }}[] = { {{ node.init_values | join(sep=", ") }} };
{%- else -%}
{{ node.c_type }} buffer_{{ node.id }}[{{ node.size }}];
{%- endif %}
{% endfor %}

/* --- Execution Graph --- */
void execute() {
{%- for group in groups %}

    /* --- Fusion Group (Shape: {{ group.shape }}) --- */
{{ group.loops_open -}}
{%- for op in group.operations %}
    {{ group.indent }}// {{ op.id }}
    {{ group.indent }}{{ op.body }}
{%- endfor %}
{{ group.loops_close -}}
{%- endfor %}
}

int main() {
    execute();

#ifdef buffer_out_SIZE
    printf("Calculation finished. Results (out): ");
    for(int i = 0; i < (buffer_out_SIZE < 10 ? buffer_out_SIZE : 10); ++i) {
        printf("%f ", (double)buffer_out[i]);
    }
    printf("\n");
#else
    printf("Calculation finished.\n");
#endif

    return 0;
}