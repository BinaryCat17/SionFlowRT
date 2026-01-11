#include <SDL2/SDL.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>

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

extern void execute_all();

{% for node in nodes -%}
extern float buffer_{{ node.prog_id }}_{{ node.node_id }} [];
{% endfor %}

{% for prog in programs %}
{% for out in prog.outputs -%}
#define buffer_{{ prog.id }}_{{ out.alias }} buffer_{{ prog.id }}_{{ out.real_id }}
{% endfor %}
{% endfor %}

int main(int argc, char* argv[]) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) return 1;

    SDL_Window* window = SDL_CreateWindow("SionFlow Painter", 
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 
        WIDTH, HEIGHT, SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);

    if (!window) return 1;

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);

    bool quit = false;
    SDL_Event e;
    uint32_t start_time = SDL_GetTicks();
    
    static float cur_x = 0.5f, cur_y = 0.5f;
    static float lst_x = 0.5f, lst_y = 0.5f;
    static bool lmb = false;

    while (!quit) {
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT) quit = true;
            if (e.type == SDL_MOUSEMOTION) {
                cur_x = (float)e.motion.x / (float)WIDTH;
                cur_y = (float)e.motion.y / (float)HEIGHT;
            }
            if (e.type == SDL_MOUSEBUTTONDOWN) {
                if (e.button.button == SDL_BUTTON_LEFT) {
                    lmb = true;
                    cur_x = (float)e.button.x / (float)WIDTH;
                    cur_y = (float)e.button.y / (float)HEIGHT;
                    lst_x = cur_x; lst_y = cur_y;
                }
            }
            if (e.type == SDL_MOUSEBUTTONUP) {
                if (e.button.button == SDL_BUTTON_LEFT) lmb = false;
            }
        }

        // --- Handle System Inputs (Sources) ---
        {% for prog in programs %}
        {% for binding in prog.link_plan.bindings %}
        {% if binding.source_type == "MousePosition" -%}
        buffer_{{ prog.id }}_{{ binding.input_node_id }}[0] = cur_x;
        buffer_{{ prog.id }}_{{ binding.input_node_id }}[1] = cur_y;
        {% elif binding.source_type == "MousePositionPrev" -%}
        buffer_{{ prog.id }}_{{ binding.input_node_id }}[0] = lst_x;
        buffer_{{ prog.id }}_{{ binding.input_node_id }}[1] = lst_y;
        {% elif binding.source_type == "MouseButton" -%}
        buffer_{{ prog.id }}_{{ binding.input_node_id }}[0] = lmb ? 1.0f : 0.0f;
        {% elif binding.source_type == "Time" -%}
        buffer_{{ prog.id }}_{{ binding.input_node_id }}[0] = (SDL_GetTicks() - start_time) / 1000.0f;
        {% elif binding.source_type == "ScreenUV" -%}
        PARALLEL
        for(int y = 0; y < HEIGHT; ++y) {
            for(int x = 0; x < WIDTH; ++x) {
                int idx = (y * WIDTH + x) * 2;
                buffer_{{ prog.id }}_{{ binding.input_node_id }}[idx + 0] = (float)x / (float)WIDTH;
                buffer_{{ prog.id }}_{{ binding.input_node_id }}[idx + 1] = (float)y / (float)HEIGHT;
            }
        }
        {%- endif %}
        {% endfor %}
        {% endfor %}

        // --- Handle Inter-program Links ---
        {% for prog in programs %}
        {% for link in prog.link_plan.inter_links %}
        memcpy(buffer_{{ link.dst_prog }}_{{ link.dst_node }}, 
               buffer_{{ link.src_prog }}_{{ link.src_node }}, 
               sizeof(float) * ({{ nodes_map[link.src_prog][link.src_node].size }}));
        {% endfor %}
        {% endfor %}

        execute_all();

        lst_x = cur_x;
        lst_y = cur_y;

        // --- Render to Display ---
        {% for prog in programs %}
        {% if prog.link_plan.display_source %}
        {% set display = prog.link_plan.display_source %}
        void* pixels; int pitch;
        SDL_LockTexture(texture, NULL, &pixels, &pitch);
        PARALLEL
        for (int y = 0; y < HEIGHT; ++y) {
            uint32_t* row = (uint32_t*)((uint8_t*)pixels + y * pitch);
            for (int x = 0; x < WIDTH; ++x) {
                int i = y * WIDTH + x;
                float r = buffer_{{ display.src_prog }}_{{ display.src_node }}[i * 4 + 0];
                float g = buffer_{{ display.src_prog }}_{{ display.src_node }}[i * 4 + 1];
                float b = buffer_{{ display.src_prog }}_{{ display.src_node }}[i * 4 + 2];
                row[x] = (255u << 24) | (((uint8_t)(fmaxf(0.0f, fminf(r, 1.0f))*255)) << 16) | (((uint8_t)(fmaxf(0.0f, fminf(g, 1.0f))*255)) << 8) | ((uint8_t)(fmaxf(0.0f, fminf(b, 1.0f))*255));
            }
        }
        SDL_UnlockTexture(texture);
        {% endif %}
        {% endfor %}

        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);
    }

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
