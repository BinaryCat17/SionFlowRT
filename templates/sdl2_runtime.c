#include <SDL2/SDL.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

/* --- Parameters --- */
{% for name, value in parameters -%}
#define {{ name }} {{ value }}
{% endfor %}

extern void execute();

{% for node in nodes -%}
extern {{ node.c_type }} buffer_{{ node.prog_id }}_{{ node.node_id }}[];
{% if node.is_stateful -%}
extern {{ node.c_type }} buffer_{{ node.prog_id }}_{{ node.node_id }}_swap[];
{%- endif %}
{% endfor %}

int main(int argc, char* argv[]) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
        return 1;
    }

    // Предполагаем, что WIDTH и HEIGHT определены в параметрах
    SDL_Window* window = SDL_CreateWindow("SionFlow SDL2 Runtime", 
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 
        WIDTH, HEIGHT, SDL_WINDOW_SHOWN);

    if (window == NULL) {
        printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, 
        SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);

    bool quit = false;
    SDL_Event e;
    uint32_t start_time = SDL_GetTicks();

    // Initial state copy for stateful nodes
    {% for node in nodes -%}{% if node.is_stateful -%}
    memcpy(buffer_{{ node.prog_id }}_{{ node.node_id }}_swap, buffer_{{ node.prog_id }}_{{ node.node_id }}, sizeof(float) * ({{ node.size_expr }})); // Hack: assuming float
    {% endif -%}{% endfor %}

    while (!quit) {
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT) quit = true;
            
            {% for m in mappings -%}
            {% if m.source.type == "MousePosition" -%}
            if (e.type == SDL_MOUSEMOTION) {
                buffer_{{ m.program }}_{{ m.tensor }}[0] = (float)e.motion.x / (float)WIDTH;
                buffer_{{ m.program }}_{{ m.tensor }}[1] = (float)e.motion.y / (float)HEIGHT;
            }
            {% elif m.source.type == "MouseButton" -%}
            if (e.type == SDL_MOUSEBUTTONDOWN || e.type == SDL_MOUSEBUTTONUP) {
                float val = (e.type == SDL_MOUSEBUTTONDOWN) ? 1.0f : 0.0f;
                {% if m.source.button == "left" -%}
                if (e.button.button == SDL_BUTTON_LEFT) buffer_{{ m.program }}_{{ m.tensor }}[0] = val;
                {% elif m.source.button == "right" -%}
                if (e.button.button == SDL_BUTTON_RIGHT) buffer_{{ m.program }}_{{ m.tensor }}[0] = val;
                {%- endif %}
            }
            {% endif -%}
            {%- endfor %}
        }

        // Global inputs
        float current_time = (SDL_GetTicks() - start_time) / 1000.0f;
        {% for m in mappings -%}
        {% if m.source.type == "Time" -%}
        buffer_{{ m.program }}_{{ m.tensor }}[0] = current_time;
        {% elif m.source.type == "ScreenUV" -%}
        for(int y = 0; y < HEIGHT; ++y) {
            for(int x = 0; x < WIDTH; ++x) {
                int idx = (y * WIDTH + x) * 2;
                buffer_{{ m.program }}_{{ m.tensor }}[idx + 0] = (float)x / (float)WIDTH;
                buffer_{{ m.program }}_{{ m.tensor }}[idx + 1] = (float)y / (float)HEIGHT;
            }
        }
        {%- endif %}{%- endfor %}

        // Resolve Links (External and Stateful)
        {% for m in mappings -%}
        {% if m.source.type == "Link" -%}
        {% if m.program == m.source.program -%}
        // Feedback loop: use swap buffer
        // Note: size calculation needs to be more robust in real codegen
        memcpy(buffer_{{ m.program }}_{{ m.tensor }}, buffer_{{ m.source.program }}_{{ m.source.output }}_swap, sizeof(float) * ({{ nodes_map[m.program][m.tensor].size_expr }}));
        {% else -%}
        // Inter-program link
        memcpy(buffer_{{ m.program }}_{{ m.tensor }}, buffer_{{ m.source.program }}_{{ m.source.output }}, sizeof(float) * ({{ nodes_map[m.program][m.tensor].size_expr }}));
        {%- endif %}{%- endif %}{%- endfor %}

        execute();

        // Update stateful swaps
        {% for node in nodes -%}{% if node.is_stateful -%}
        memcpy(buffer_{{ node.prog_id }}_{{ node.node_id }}_swap, buffer_{{ node.prog_id }}_{{ node.node_id }}, sizeof(float) * ({{ node.size_expr }}));
        {% endif -%}{% endfor %}

        // Render Sinks
        {% for m in mappings -%}
        {% if m.source.type == "Display" -%}
        void* pixels; int pitch;
        SDL_LockTexture(texture, NULL, &pixels, &pitch);
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
        {% endif -%}{%- endfor %}

        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);

        static int frame_counter = 0;
        if (frame_counter == 60) {
            int sw = WIDTH / 2;
            int sh = HEIGHT / 2;
            SDL_Surface* full_surface = SDL_CreateRGBSurface(0, WIDTH, HEIGHT, 32, 0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000);
            SDL_Surface* small_surface = SDL_CreateRGBSurface(0, sw, sh, 32, 0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000);
            if (full_surface && small_surface) {
                SDL_RenderReadPixels(renderer, NULL, full_surface->format->format, full_surface->pixels, full_surface->pitch);
                SDL_BlitScaled(full_surface, NULL, small_surface, NULL);
                if (SDL_SaveBMP(small_surface, "logs/screenshot.bmp") == 0) {
                    printf("Half-size screenshot saved to logs/screenshot.bmp at frame 60\n");
                } else {
                    printf("Failed to save screenshot: %s\n", SDL_GetError());
                }
            }
            if (full_surface) SDL_FreeSurface(full_surface);
            if (small_surface) SDL_FreeSurface(small_surface);
        }
        if (frame_counter <= 60) frame_counter++;
    }

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
