// warshall_menu.c
// CPU secuencial: Cerradura transitiva booleana (Warshall lógico)
// + MENÚ interactivo:
//   (1) ingresar matriz manual + parámetros
//   (2) ingresar parámetros + grafo random
//   (3) grafo y parámetros random
//
// Mantiene modo clásico por argumentos:
//   ./warshall N p seed repeats verify [print]
//
// Compilar:
//   gcc -O3 -std=c11 warshall_menu.c -o warshall
// (si tu Linux requiere):
//   gcc -O3 -std=c11 warshall_menu.c -o warshall -lrt

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <ctype.h>

static inline double seconds_now(void) {
    struct timespec ts;
#if defined(CLOCK_MONOTONIC)
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }
#else
    if (clock_gettime(CLOCK_REALTIME, &ts) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }
#endif
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint8_t* alloc_matrix(int N) {
    size_t bytes = (size_t)N * (size_t)N * sizeof(uint8_t);
    uint8_t* A = (uint8_t*)malloc(bytes);
    if (!A) {
        fprintf(stderr, "ERROR: no se pudo asignar %zu bytes para N=%d\n", bytes, N);
        exit(EXIT_FAILURE);
    }
    return A;
}

static void init_random(uint8_t* A, int N, double p, unsigned seed) {
    srand(seed);
    for (int i = 0; i < N * N; i++) {
        double r = (double)rand() / (double)RAND_MAX;
        A[i] = (r < p) ? 1 : 0;
    }
}

static void print_matrix(const uint8_t* A, int N, const char* title) {
    printf("\n=== %s (N=%d) ===\n", title, N);

    printf("     ");
    for (int j = 0; j < N; j++) printf("%2d ", j);
    printf("\n");

    printf("     ");
    for (int j = 0; j < N; j++) printf("---");
    printf("\n");

    for (int i = 0; i < N; i++) {
        printf("%2d | ", i);
        const uint8_t* row = &A[i * N];
        for (int j = 0; j < N; j++) printf("%2d ", (int)row[j]);
        printf("\n");
    }
}

// -------------------------
// Núcleo: Warshall lógico
// -------------------------
void warshall_logical(uint8_t* A, int N) {
    // A[i][j] = A[i][j] OR (A[i][k] AND A[k][j])
    // NO fuerza diagonal a 1.
    for (int k = 0; k < N; k++) {
        const uint8_t* row_k = &A[k * N];
        for (int i = 0; i < N; i++) {
            uint8_t aik = A[i * N + k];
            if (!aik) continue;
            uint8_t* row_i = &A[i * N];
            for (int j = 0; j < N; j++) {
                row_i[j] = (uint8_t)(row_i[j] | (aik & row_k[j]));
            }
        }
    }
}

// -----------------------------------
// Referencia: cerradura transitiva con BFS
// -----------------------------------
static void bfs_closure_ref(const uint8_t* Ain, uint8_t* Rref, int N) {
    int* queue = (int*)malloc((size_t)N * sizeof(int));
    uint8_t* vis = (uint8_t*)malloc((size_t)N * sizeof(uint8_t));
    if (!queue || !vis) {
        fprintf(stderr, "ERROR: memoria insuficiente para BFS\n");
        free(queue);
        free(vis);
        exit(EXIT_FAILURE);
    }

    for (int s = 0; s < N; s++) {
        memset(vis, 0, (size_t)N);
        int front = 0, back = 0;

        const uint8_t* row_s = &Ain[s * N];
        for (int v = 0; v < N; v++) {
            if (row_s[v]) {
                vis[v] = 1;
                queue[back++] = v;
            }
        }

        while (front < back) {
            int u = queue[front++];
            const uint8_t* row_u = &Ain[u * N];
            for (int v = 0; v < N; v++) {
                if (row_u[v] && !vis[v]) {
                    vis[v] = 1;
                    queue[back++] = v;
                }
            }
        }

        uint8_t* row_ref = &Rref[s * N];
        for (int j = 0; j < N; j++) row_ref[j] = vis[j];
    }

    free(queue);
    free(vis);
}

static int verify_against_ref(const uint8_t* Rref, const uint8_t* Aout, int N) {
    for (int i = 0; i < N; i++) {
        const uint8_t* rr = &Rref[i * N];
        const uint8_t* ao = &Aout[i * N];
        for (int j = 0; j < N; j++) {
            if (rr[j] != ao[j]) {
                fprintf(stderr,
                        "FALLO verificación: fila i=%d, col j=%d | esperado=%d, obtenido=%d\n",
                        i, j, (int)rr[j], (int)ao[j]);
                return 0;
            }
        }
    }
    return 1;
}

// =====================================================
// Helpers de input robusto (sin scanf)
// =====================================================
static void read_line(char* buf, size_t n) {
    if (!fgets(buf, (int)n, stdin)) {
        printf("\nEOF detectado. Saliendo.\n");
        exit(0);
    }
}

static long read_long_prompt(const char* prompt, long minv, long maxv) {
    char line[256];
    for (;;) {
        printf("%s", prompt);
        read_line(line, sizeof(line));

        char* end = NULL;
        errno = 0;
        long v = strtol(line, &end, 10);
        if (errno == 0) {
            while (end && *end && isspace((unsigned char)*end)) end++;
            if (end && (*end == '\0' || *end == '\n')) {
                if (v >= minv && v <= maxv) return v;
            }
        }
        printf("Entrada inválida. Rango permitido: [%ld..%ld]\n", minv, maxv);
    }
}

static double read_double_prompt(const char* prompt, double minv, double maxv) {
    char line[256];
    for (;;) {
        printf("%s", prompt);
        read_line(line, sizeof(line));

        char* end = NULL;
        errno = 0;
        double v = strtod(line, &end);
        if (errno == 0) {
            while (end && *end && isspace((unsigned char)*end)) end++;
            if (end && (*end == '\0' || *end == '\n')) {
                if (v >= minv && v <= maxv) return v;
            }
        }
        printf("Entrada inválida. Rango permitido: [%.3f..%.3f]\n", minv, maxv);
    }
}

static int read_int_prompt(const char* prompt, int minv, int maxv) {
    return (int)read_long_prompt(prompt, (long)minv, (long)maxv);
}

static int read_yesno_prompt(const char* prompt) {
    char line[64];
    for (;;) {
        printf("%s (1=si, 0=no): ", prompt);
        read_line(line, sizeof(line));
        if (line[0] == '1') return 1;
        if (line[0] == '0') return 0;
        printf("Entrada inválida. Escribe 1 o 0.\n");
    }
}

static int parse_row_01(const char* line, uint8_t* row, int N) {
    // Acepta: "0 1 0 1" o "0101..." (con o sin espacios)
    int count = 0;
    for (const char* p = line; *p && count < N; p++) {
        if (*p == '0' || *p == '1') {
            row[count++] = (uint8_t)(*p - '0');
        }
    }
    return (count == N);
}

static double density_ones(const uint8_t* A, int N) {
    long long ones = 0;
    for (long long i = 0; i < (long long)N * (long long)N; i++) ones += A[i] ? 1 : 0;
    return (double)ones / (double)((long long)N * (long long)N);
}

// =====================================================
// Ejecutar en modo verify/timing (reusa tu lógica)
// =====================================================
static int run_experiment(uint8_t* Ain, int N, double p, unsigned seed, int repeats, int verify, int print) {
    const int PRINT_LIMIT = 16;

    if (N <= PRINT_LIMIT) { // comportamiento original
        verify = 1;
        print = 1;
    }
    if (repeats <= 0) repeats = 1;

    uint8_t* A = alloc_matrix(N);

    if (verify) {
        if (N > 128) {
            printf("Aviso: verificación activada con N=%d; se recomienda N<=128.\n", N);
        }

        memcpy(A, Ain, (size_t)N * (size_t)N);

        double t0 = seconds_now();
        warshall_logical(A, N);
        double t1 = seconds_now();
        double kernel_time = t1 - t0;

        uint8_t* Rref = alloc_matrix(N);
        bfs_closure_ref(Ain, Rref, N);

        int ok = verify_against_ref(Rref, A, N);

        if (print) {
            print_matrix(Ain,  N, "MATRIZ DE ENTRADA (Grafo / Adyacencia)");
            print_matrix(Rref, N, "MATRIZ DE VERIFICACIÓN (Referencia BFS)");
            print_matrix(A,    N, "MATRIZ DE SALIDA (Warshall lógico)");
        }

        printf("\nVALIDACIÓN (BFS) para N=%d: %s\n", N, ok ? "OK" : "FALLIDA");
        printf("Tiempo del núcleo (warshall_logical): %.6f s\n", kernel_time);
        printf("Resumen params | N=%d | p=%.3f | seed=%u | repeats=%d | verify=%d | print=%d\n",
               N, p, seed, repeats, verify, print);

        free(Rref);
        free(A);
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    double best = 1e100;
    for (int r = 0; r < repeats; r++) {
        memcpy(A, Ain, (size_t)N * (size_t)N);
        double t0 = seconds_now();
        warshall_logical(A, N);
        double t1 = seconds_now();
        double dt = t1 - t0;
        if (dt < best) best = dt;
    }

    printf("CPU Warshall lógico | N=%d | p=%.3f | seed=%u | repeats=%d | best_kernel_time=%.6f s\n",
           N, p, seed, repeats, best);

    free(A);
    return EXIT_SUCCESS;
}

// =====================================================
// Menú
// =====================================================
static void menu_loop(void) {
    for (;;) {
        printf("\n==============================\n");
        printf("   MENÚ - Warshall lógico CPU\n");
        printf("==============================\n");
        printf("1) Ingresar MATRIZ manual + parámetros\n");
        printf("2) Ingresar parámetros + GRAFO random\n");
        printf("3) Grafo y parámetros RANDOM\n");
        printf("0) Salir\n");

        int opt = read_int_prompt("Opción: ", 0, 3);
        if (opt == 0) break;

        int N = 256;
        double p = 0.05;
        unsigned seed = 1234;
        int repeats = 3;
        int verify = 0;
        int print = 0;

        uint8_t* Ain = NULL;

        if (opt == 1) {
            N = read_int_prompt("Ingrese N (1..2048 recomendado): ", 1, 4096);
            Ain = alloc_matrix(N);

            printf("\nIngrese la matriz de adyacencia (%dx%d) con 0/1.\n", N, N);
            printf("Formato permitido por fila: '0 1 0 1' o '0101...'\n\n");

            char line[8192];
            for (int i = 0; i < N; i++) {
                for (;;) {
                    printf("Fila %d: ", i);
                    read_line(line, sizeof(line));
                    if (parse_row_01(line, &Ain[i * N], N)) break;
                    printf("Fila inválida. Debe contener %d valores 0/1.\n", N);
                }
            }

            // p se calcula como densidad real (informativo)
            p = density_ones(Ain, N);
            seed = 0;

            repeats = read_int_prompt("repeats (>=1): ", 1, 1000);
            verify  = read_yesno_prompt("verify");
            print   = read_yesno_prompt("print");

            printf("\nDensidad p calculada desde la matriz: %.3f\n", p);
        }
        else if (opt == 2) {
            N = read_int_prompt("N (1..4096): ", 1, 4096);
            p = read_double_prompt("p (0..1): ", 0.0, 1.0);
            seed = (unsigned)read_long_prompt("seed (0..2^31-1): ", 0, 2147483647L);
            repeats = read_int_prompt("repeats (>=1): ", 1, 1000);
            verify  = read_yesno_prompt("verify");
            print   = read_yesno_prompt("print");

            Ain = alloc_matrix(N);
            init_random(Ain, N, p, seed);
        }
        else if (opt == 3) {
            // Random razonable (puedes ajustar los rangos)
            unsigned s = (unsigned)time(NULL);
            srand(s);

            int choices[] = {8,16,32,64,128,256,512,1024};
            int idx = rand() % (int)(sizeof(choices)/sizeof(choices[0]));
            N = choices[idx];

            p = 0.01 + ((double)rand() / (double)RAND_MAX) * 0.19; // [0.01..0.20]
            seed = (unsigned)rand();
            repeats = 1 + (rand() % 7); // 1..7

            // si N pequeño, validamos; si no, solo timing
            verify = (N <= 128) ? 1 : 0;
            print  = (N <= 16) ? 1 : 0;

            Ain = alloc_matrix(N);
            init_random(Ain, N, p, seed);

            printf("\nParámetros random generados:\n");
            printf("N=%d | p=%.3f | seed=%u | repeats=%d | verify=%d | print=%d\n",
                   N, p, seed, repeats, verify, print);
        }

        int rc = run_experiment(Ain, N, p, seed, repeats, verify, print);
        free(Ain);

        if (rc != EXIT_SUCCESS) {
            printf("Ejecución terminó con error (validación fallida o problema).\n");
        }

        if (!read_yesno_prompt("\n¿Deseas ejecutar otra vez?")) break;
    }
}

int main(int argc, char** argv) {
    // --- Modo por argumentos (tu modo original) ---
    if (argc >= 2) {
        int N = 256;
        double p = 0.05;
        unsigned seed = 1234;
        int repeats = 3;
        int verify = 0;
        int print = 0;

        if (argc >= 2) N = atoi(argv[1]);
        if (argc >= 3) p = atof(argv[2]);
        if (argc >= 4) seed = (unsigned)atoi(argv[3]);
        if (argc >= 5) repeats = atoi(argv[4]);
        if (argc >= 6) verify = atoi(argv[5]);
        if (argc >= 7) print = atoi(argv[6]);

        if (N <= 0) { fprintf(stderr, "ERROR: N debe ser > 0\n"); return EXIT_FAILURE; }
        if (p < 0.0 || p > 1.0) { fprintf(stderr, "ERROR: p debe estar en [0,1]\n"); return EXIT_FAILURE; }
        if (repeats <= 0) repeats = 1;

        const int PRINT_LIMIT = 16;
        if (N <= PRINT_LIMIT) { verify = 1; print = 1; }

        uint8_t* Ain = alloc_matrix(N);
        init_random(Ain, N, p, seed);

        int rc = run_experiment(Ain, N, p, seed, repeats, verify, print);
        free(Ain);
        return rc;
    }

    // --- Modo menú ---
    menu_loop();
    return 0;
}
