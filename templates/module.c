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

/* --- Global Resources --- */
{% for id, res in orchestration.resources -%}
{% set size = res.shape | join(sep=" * ") -%}
{% if res.is_state -%}
float resource_{{ id }}_front[{{ size }}];
float resource_{{ id }}_back[{{ size }}];
float* resource_{{ id }} = resource_{{ id }}_front;
{% else -%}
float resource_{{ id }}_data[{{ size }}];
float* resource_{{ id }} = resource_{{ id }}_data;
{% endif -%}
{% endfor %}

/* --- Tensor Buffers --- */
{% for prog in programs -%}
/* Program: {{ prog.id }} */
{% for group in prog.loop_groups -%}
{% for node in group.nodes -%}
float buffer_{{ prog.id }}_{{ node.id }}[{{ group.size }}];
{% endfor -%}
{% endfor -%}
{% for out in prog.outputs -%}
#define buffer_{{ prog.id }}_{{ out.alias }} buffer_{{ prog.id }}_{{ out.real_id }}
{% endfor %}
{% endfor %}

/* --- Execution Functions --- */
{% for prog in programs -%}
void execute_{{ prog.id }}() {
    // 1. Copy Resources to Inputs
    {% for binding in prog.instance.inputs -%}
    {% set res = orchestration.resources[binding.resource_id] -%}
    {% set size = res.shape | join(sep=" * ") -%}
    memcpy(buffer_{{ prog.id }}_{{ binding.program_port }}, resource_{{ binding.resource_id }}, sizeof(float) * ({{ size }}));
    {% endfor %}

    // 2. Run Program Logic
{%- for group in prog.loop_groups -%}
    {%- if group.has_body -%}
        {%- if group.is_fusible %}
    // Fused Loop (Size: {{ group.size }})
    PARALLEL
    for(int i = 0; i < {{ group.size }}; ++i) {
        {%- for node in group.nodes -%}
            {%- if node.body != "" %}
        {{ node.body }}
            {%- endif -%}
        {%- endfor %}
    }
        {%- else -%}
            {%- for node in group.nodes -%}
                {%- if node.body != "" %}
    // Single Node: {{ node.id }}
    for(int i = 0; i < {{ group.size }}; ++i) {
        {{ node.body }}
    }
                {%- endif -%}
            {%- endfor -%}
        {%- endif -%}
    {%- endif -%}
{%- endfor %}

    // 3. Copy Outputs back to Resources
    {% for binding in prog.instance.outputs -%}
    {% set res = orchestration.resources[binding.resource_id] -%}
    {% set size = res.shape | join(sep=" * ") -%}
    {% if res.is_state -%}
    memcpy(resource_{{ binding.resource_id }}_back, buffer_{{ prog.id }}_{{ binding.program_port }}, sizeof(float) * ({{ size }}));
    {% else -%}
    memcpy(resource_{{ binding.resource_id }}, buffer_{{ prog.id }}_{{ binding.program_port }}, sizeof(float) * ({{ size }}));
    {% endif -%}
    {% endfor %}
}
{% endfor %}

void swap_states() {
    {% for id, res in orchestration.resources -%}
    {% if res.is_state -%}
    {% set size = res.shape | join(sep=" * ") -%}
    memcpy(resource_{{ id }}_front, resource_{{ id }}_back, sizeof(float) * ({{ size }}));
    {% endif -%}
    {% endfor %}
}

void execute_all() {
{%- for prog in programs %}
    execute_{{ prog.id }}();
{%- endfor %}
    swap_states();
}