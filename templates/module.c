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
{% for node in nodes -%}
#define buffer_{{ node.prog_id }}_{{ node.node_id }}_SIZE ({{ node.size_expr }})
{% if node.init_values -%}
{{ node.c_type }} buffer_{{ node.prog_id }}_{{ node.node_id }}[] = { {{ node.init_values | join(sep=", ") }} };
{%- else -%}
{{ node.c_type }} buffer_{{ node.prog_id }}_{{ node.node_id }}[buffer_{{ node.prog_id }}_{{ node.node_id }}_SIZE];
{%- endif %}
{% if node.is_stateful -%}
{{ node.c_type }} buffer_{{ node.prog_id }}_{{ node.node_id }}_swap[buffer_{{ node.prog_id }}_{{ node.node_id }}_SIZE];
{%- endif %}
{% endfor %}

/* --- Execution Graph --- */
void execute() {
{%- for group in groups %}

    /* --- Fusion Group (Prog: {{ group.prog_id }}, Shape: {{ group.shape }}) --- */
{{ group.loops_open -}}
{%- for op in group.operations %}
    {{ group.indent }}// {{ op.id }}
    {{ group.indent }}{{ op.body }}
{%- endfor %}
{{ group.loops_close -}}
{%- endfor %}
}
