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

        // Inputs from mappings
        {% for m in mappings -%}
        {% if m.source.type == "MousePosition" -%}
        buffer_{{ m.program }}_{{ m.tensor }}[0] = cur_x;
        buffer_{{ m.program }}_{{ m.tensor }}[1] = cur_y;
        {% elif m.source.type == "MousePositionPrev" -%}
        buffer_{{ m.program }}_{{ m.tensor }}[0] = lst_x;
        buffer_{{ m.program }}_{{ m.tensor }}[1] = lst_y;
        {% elif m.source.type == "MouseButton" -%}
        buffer_{{ m.program }}_{{ m.tensor }}[0] = lmb ? 1.0f : 0.0f;
        {% elif m.source.type == "Time" -%}
        buffer_{{ m.program }}_{{ m.tensor }}[0] = (SDL_GetTicks() - start_time) / 1000.0f;
        {% elif m.source.type == "ScreenUV" -%}
        PARALLEL
        for(int y = 0; y < HEIGHT; ++y) {
            for(int x = 0; x < WIDTH; ++x) {
                int idx = (y * WIDTH + x) * 2;
                buffer_{{ m.program }}_{{ m.tensor }}[idx + 0] = (float)x / (float)WIDTH;
                buffer_{{ m.program }}_{{ m.tensor }}[idx + 1] = (float)y / (float)HEIGHT;
            }
        }
        {%- endif %}{%- endfor %}

        // Links
        {% for m in mappings -%}{% if m.source.type == "Link" -%}
        memcpy(buffer_{{ m.program }}_{{ m.tensor }}, buffer_{{ m.source.program }}_{{ m.source.output }}, sizeof(float) * ({{ nodes_map[m.source.program][m.source.output].size }}));
        {%- endif %}{%- endfor %}

        execute_all();

        lst_x = cur_x;
        lst_y = cur_y;

        // Render to display
        {% for m in mappings -%}{% if m.source.type == "Display" -%}
        void* pixels; int pitch;
        SDL_LockTexture(texture, NULL, &pixels, &pitch);
        PARALLEL
        for (int y = 0; y < HEIGHT; ++y) {
            uint32_t* row = (uint32_t*)((uint8_t*)pixels + y * pitch);
            for (int x = 0; x < WIDTH; ++x) {
                int i = y * WIDTH + x;
                float r = buffer_{{ m.program }}_{{ m.tensor }}[i * 4 + 0];
                float g = buffer_{{ m.program }}_{{ m.tensor }}[i * 4 + 1];
                float b = buffer_{{ m.program }}_{{ m.tensor }}[i * 4 + 2];
                row[x] = (255u << 24) | (((uint8_t)(fmaxf(0.0f, fminf(r, 1.0f))*255)) << 16) | (((uint8_t)(fmaxf(0.0f, fminf(g, 1.0f))*255)) << 8) | ((uint8_t)(fmaxf(0.0f, fminf(b, 1.0f))*255));
            }
        }
        SDL_UnlockTexture(texture);
        {% endif %}{%- endfor %}

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
