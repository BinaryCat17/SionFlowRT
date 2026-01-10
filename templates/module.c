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

    /* --- Group (Prog: {{ group.prog_id }}, Shape: {{ group.shape }}) --- */
{%- if group.is_parallel %}
    PARALLEL
{%- endif %}
{%- for lp in group.outer_loops %}
    {% for i in range(end=loop.index0) %}    {% endfor %}for(int {{ lp.var }} = 0; {{ lp.var }} < {{ lp.limit }}; ++{{ lp.var }}) {
{%- endfor %}

    {%- set outer_rank = group.outer_loops | length %}
    {%- set base_indent = "" %}
    {%- for i in range(end=outer_rank) %}{% set_global base_indent = base_indent ~ "    " %}{% endfor %}

    {%- if group.kernel %}
    {{ base_indent }}{{ group.kernel.init }}
    {%- for lp in group.kernel.inner_loops %}
    {{ base_indent }}{% for i in range(end=loop.index0) %}    {% endfor %}for(int {{ lp.var }} = 0; {{ lp.var }} < {{ lp.limit }}; ++{{ lp.var }}) {
    {%- endfor %}
    
    {%- set inner_rank = group.kernel.inner_loops | length %}
    {%- set body_indent = base_indent %}
    {%- for i in range(end=inner_rank) %}{% set_global body_indent = body_indent ~ "    " %}{% endfor %}
    {{ body_indent }}{{ group.kernel.body }}

    {%- for lp in group.kernel.inner_loops %}
    {{ base_indent }}{% for i in range(end=(inner_rank - loop.index)) %}    {% endfor %}}
    {%- endfor %}
    {{ base_indent }}{{ group.kernel.finalize }}

    {%- else %}
    {%- for op in group.fusion_ops %}
    {{ base_indent }}// {{ op.id }}
    {{ base_indent }}{{ op.body }}
    {%- endfor %}
    {%- endif %}

{%- for lp in group.outer_loops %}
    {% for i in range(end=(outer_rank - loop.index)) %}    {% endfor %}}
{%- endfor %}
{%- endfor %}
}
