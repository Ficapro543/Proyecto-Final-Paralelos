// warshall_menu_cuda.cu
// CUDA: Cerradura transitiva booleana (Warshall logico) + MENU interactivo
//
// Opciones de menu:
//   (1) ingresar matriz manual + parametros
//   (2) ingresar parametros + grafo random
//   (3) grafo y parametros random
//
// Modo por argumentos (como antes):
//   ./warshall_cuda N p seed repeats verify [print]
//
// Compilar (Linux):
//   nvcc -O3 -std=c++17 warshall_menu_cuda.cu -o warshall_cuda
// (si tu Linux requiere -lrt):
//   nvcc -O3 -std=c++17 warshall_menu_cuda.cu -o warshall_cuda -lrt
//
// Ejecutar:
//   ./warshall_cuda
//   ./warshall_cuda 1024 0.05 1234 3 0 0
//
// Nota de paralelizacion:
// - k se mantiene SECUENCIAL (dependencia entre iteraciones).
// - Para cada k, se lanza un kernel 2D que actualiza TODAS las celdas (i,j) en paralelo.
//
// IMPORTANTE (correctitud):
// - Este kernel lee A[i,k] y A[k,j] del MISMO buffer A que tambien se escribe.
//   Eso replica tu version CUDA didactica anterior, y suele funcionar para Warshall booleano,
//   pero si quieres "paso k limpio" (lectura desde snapshot), usa doble buffer (Ain->Aout)
//   por cada k (mas lento por copias). Aqui dejo la version simple y rapida (in-place).

#define _POSIX_C_SOURCE 200809L

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <cerrno>
#include <cctype>
#include <iostream>

static inline void CUDA_CHECK(cudaError_t e, const char* file, int line) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", file, line, cudaGetErrorString(e));
        std::exit(EXIT_FAILURE);
    }
}
#define CUDA_CALL(x) CUDA_CHECK((x), __FILE__, __LINE__)

// =====================================================
// Timing (CPU wall time) para medir total del "nucleo"
// =====================================================
static inline double seconds_now(void) {
    struct timespec ts;
#if defined(CLOCK_MONOTONIC)
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        perror("clock_gettime");
        std::exit(EXIT_FAILURE);
    }
#else
    if (clock_gettime(CLOCK_REALTIME, &ts) != 0) {
        perror("clock_gettime");
        std::exit(EXIT_FAILURE);
    }
#endif
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// =====================================================
// Memoria / utilidades
// =====================================================
static uint8_t* alloc_matrix(int N) {
    size_t bytes = (size_t)N * (size_t)N * sizeof(uint8_t);
    uint8_t* A = (uint8_t*)std::malloc(bytes);
    if (!A) {
        std::fprintf(stderr, "ERROR: no se pudo asignar %zu bytes para N=%d\n", bytes, N);
        std::exit(EXIT_FAILURE);
    }
    return A;
}

static void init_random(uint8_t* A, int N, double p, unsigned seed) {
    std::srand(seed);
    for (int i = 0; i < N * N; i++) {
        double r = (double)std::rand() / (double)RAND_MAX;
        A[i] = (r < p) ? 1 : 0;
    }
}

static void print_matrix(const uint8_t* A, int N, const char* title) {
    std::printf("\n=== %s (N=%d) ===\n", title, N);

    std::printf("     ");
    for (int j = 0; j < N; j++) std::printf("%2d ", j);
    std::printf("\n");

    std::printf("     ");
    for (int j = 0; j < N; j++) std::printf("---");
    std::printf("\n");

    for (int i = 0; i < N; i++) {
        std::printf("%2d | ", i);
        const uint8_t* row = &A[i * N];
        for (int j = 0; j < N; j++) std::printf("%2d ", (int)row[j]);
        std::printf("\n");
    }
}

// =====================================================
// CUDA kernel: 1 paso k (in-place)
// =====================================================
__global__ void warshall_step_u8(uint8_t* A, int N, int k) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; // col
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row
    if (i < N && j < N) {
        uint8_t aik = A[i * N + k];
        if (!aik) return; // pequeno atajo
        uint8_t akj = A[k * N + j];
        uint8_t aij = A[i * N + j];
        A[i * N + j] = (uint8_t)(aij | (aik & akj));
    }
}

// =====================================================
// Nucleo: Warshall logico (CUDA)
// =====================================================
static void warshall_logical_cuda(uint8_t* A_host, int N) {
    size_t bytes = (size_t)N * (size_t)N * sizeof(uint8_t);

    uint8_t* d_A = nullptr;
    CUDA_CALL(cudaMalloc((void**)&d_A, bytes));
    CUDA_CALL(cudaMemcpy(d_A, A_host, bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);

    for (int k = 0; k < N; k++) {
        warshall_step_u8<<<grid, block>>>(d_A, N, k);
        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize()); // claridad/didactica
    }

    CUDA_CALL(cudaMemcpy(A_host, d_A, bytes, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(d_A));
}

// -----------------------------------
// Referencia: cerradura transitiva con BFS (CPU)
// -----------------------------------
static void bfs_closure_ref(const uint8_t* Ain, uint8_t* Rref, int N) {
    int* queue = (int*)std::malloc((size_t)N * sizeof(int));
    uint8_t* vis = (uint8_t*)std::malloc((size_t)N * sizeof(uint8_t));
    if (!queue || !vis) {
        std::fprintf(stderr, "ERROR: memoria insuficiente para BFS\n");
        std::free(queue);
        std::free(vis);
        std::exit(EXIT_FAILURE);
    }

    for (int s = 0; s < N; s++) {
        std::memset(vis, 0, (size_t)N);
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

    std::free(queue);
    std::free(vis);
}

static int verify_against_ref(const uint8_t* Rref, const uint8_t* Aout, int N) {
    for (int i = 0; i < N; i++) {
        const uint8_t* rr = &Rref[i * N];
        const uint8_t* ao = &Aout[i * N];
        for (int j = 0; j < N; j++) {
            if (rr[j] != ao[j]) {
                std::fprintf(stderr,
                        "FALLO verificacion: fila i=%d, col j=%d | esperado=%d, obtenido=%d\n",
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
    if (!std::fgets(buf, (int)n, stdin)) {
        std::printf("\nEOF detectado. Saliendo.\n");
        std::exit(0);
    }
}

static long read_long_prompt(const char* prompt, long minv, long maxv) {
    char line[256];
    for (;;) {
        std::printf("%s", prompt);
        read_line(line, sizeof(line));

        char* end = NULL;
        errno = 0;
        long v = std::strtol(line, &end, 10);
        if (errno == 0) {
            while (end && *end && std::isspace((unsigned char)*end)) end++;
            if (end && (*end == '\0' || *end == '\n')) {
                if (v >= minv && v <= maxv) return v;
            }
        }
        std::printf("Entrada invalida. Rango permitido: [%ld..%ld]\n", minv, maxv);
    }
}

static double read_double_prompt(const char* prompt, double minv, double maxv) {
    char line[256];
    for (;;) {
        std::printf("%s", prompt);
        read_line(line, sizeof(line));

        char* end = NULL;
        errno = 0;
        double v = std::strtod(line, &end);
        if (errno == 0) {
            while (end && *end && std::isspace((unsigned char)*end)) end++;
            if (end && (*end == '\0' || *end == '\n')) {
                if (v >= minv && v <= maxv) return v;
            }
        }
        std::printf("Entrada invalida. Rango permitido: [%.3f..%.3f]\n", minv, maxv);
    }
}

static int read_int_prompt(const char* prompt, int minv, int maxv) {
    return (int)read_long_prompt(prompt, (long)minv, (long)maxv);
}

static int read_yesno_prompt(const char* prompt) {
    char line[64];
    for (;;) {
        std::printf("%s (1=si, 0=no): ", prompt);
        read_line(line, sizeof(line));
        if (line[0] == '1') return 1;
        if (line[0] == '0') return 0;
        std::printf("Entrada invalida. Escribe 1 o 0.\n");
    }
}

static int parse_row_01(const char* line, uint8_t* row, int N) {
    int count = 0;
    for (const char* p = line; *p && count < N; p++) {
        if (*p == '0' || *p == '1') row[count++] = (uint8_t)(*p - '0');
    }
    return (count == N);
}

static double density_ones(const uint8_t* A, int N) {
    long long ones = 0;
    for (long long i = 0; i < (long long)N * (long long)N; i++) ones += A[i] ? 1 : 0;
    return (double)ones / (double)((long long)N * (long long)N);
}

// =====================================================
// Ejecutar en modo verify/timing (CUDA)
// =====================================================
static int run_experiment(uint8_t* Ain, int N, double p, unsigned seed, int repeats, int verify, int print) {
    const int PRINT_LIMIT = 16;

    if (N <= PRINT_LIMIT) { verify = 1; print = 1; }
    if (repeats <= 0) repeats = 1;

    uint8_t* A = alloc_matrix(N);

    if (verify) {
        if (N > 128) {
            std::printf("Aviso: verificacion activada con N=%d; se recomienda N<=128.\n", N);
        }

        std::memcpy(A, Ain, (size_t)N * (size_t)N);

        double t0 = seconds_now();
        warshall_logical_cuda(A, N);
        double t1 = seconds_now();
        double kernel_time = t1 - t0;

        uint8_t* Rref = alloc_matrix(N);
        bfs_closure_ref(Ain, Rref, N);

        int ok = verify_against_ref(Rref, A, N);

        if (print) {
            print_matrix(Ain,  N, "MATRIZ DE ENTRADA (Grafo / Adyacencia)");
            print_matrix(Rref, N, "MATRIZ DE VERIFICACIoN (Referencia BFS)");
            print_matrix(A,    N, "MATRIZ DE SALIDA (Warshall logico) [CUDA]");
        }

        std::printf("\nVALIDACIoN (BFS) para N=%d: %s\n", N, ok ? "OK" : "FALLIDA");
        std::printf("Tiempo del nucleo (warshall_logical_cuda): %.6f s\n", kernel_time);
        std::printf("Resumen params | N=%d | p=%.3f | seed=%u | repeats=%d | verify=%d | print=%d\n",
                    N, p, seed, repeats, verify, print);

        std::free(Rref);
        std::free(A);
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    double best = 1e100;
    for (int r = 0; r < repeats; r++) {
        std::memcpy(A, Ain, (size_t)N * (size_t)N);
        double t0 = seconds_now();
        warshall_logical_cuda(A, N);
        double t1 = seconds_now();
        double dt = t1 - t0;
        if (dt < best) best = dt;
    }

    std::printf("CUDA Warshall logico | N=%d | p=%.3f | seed=%u | repeats=%d | best_kernel_time=%.6f s\n",
                N, p, seed, repeats, best);

    std::free(A);
    return EXIT_SUCCESS;
}

// =====================================================
// Menu
// =====================================================
static void menu_loop(void) {
    for (;;) {
        std::printf("\n==============================\n");
        std::printf("   MENu - Warshall logico CUDA\n");
        std::printf("==============================\n");
        std::printf("1) Ingresar MATRIZ manual + parametros\n");
        std::printf("2) Ingresar parametros + GRAFO random\n");
        std::printf("3) Grafo y parametros RANDOM\n");
        std::printf("0) Salir\n");

        int opt = read_int_prompt("Opcion: ", 0, 3);
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

            std::printf("\nIngrese la matriz de adyacencia (%dx%d) con 0/1.\n", N, N);
            std::printf("Formato permitido por fila: '0 1 0 1' o '0101...'\n\n");

            char line[8192];
            for (int i = 0; i < N; i++) {
                for (;;) {
                    std::printf("Fila %d: ", i);
                    read_line(line, sizeof(line));
                    if (parse_row_01(line, &Ain[i * N], N)) break;
                    std::printf("Fila invalida. Debe contener %d valores 0/1.\n", N);
                }
            }

            p = density_ones(Ain, N);
            seed = 0;

            repeats = read_int_prompt("repeats (>=1): ", 1, 1000);
            verify  = read_yesno_prompt("verify");
            print   = read_yesno_prompt("print");

            std::printf("\nDensidad p calculada desde la matriz: %.3f\n", p);
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
            unsigned s = (unsigned)time(NULL);
            srand(s);

            int choices[] = {8,16,32,64,128,256,512,1024};
            int idx = rand() % (int)(sizeof(choices)/sizeof(choices[0]));
            N = choices[idx];

            p = 0.01 + ((double)rand() / (double)RAND_MAX) * 0.19; // [0.01..0.20]
            seed = (unsigned)rand();
            repeats = 1 + (rand() % 7);

            verify = (N <= 128) ? 1 : 0;
            print  = (N <= 16) ? 1 : 0;

            Ain = alloc_matrix(N);
            init_random(Ain, N, p, seed);

            std::printf("\nParametros random generados:\n");
            std::printf("N=%d | p=%.3f | seed=%u | repeats=%d | verify=%d | print=%d\n",
                        N, p, seed, repeats, verify, print);
        }

        int rc = run_experiment(Ain, N, p, seed, repeats, verify, print);
        std::free(Ain);

        if (rc != EXIT_SUCCESS) {
            std::printf("Ejecucion termino con error (validacion fallida o problema).\n");
        }

        if (!read_yesno_prompt("\nDeseas ejecutar otra vez")) break;
    }
}

int main(int argc, char** argv) {
    // --- Modo por argumentos ---
    if (argc >= 2) {
        int N = 256;
        double p = 0.05;
        unsigned seed = 1234;
        int repeats = 3;
        int verify = 0;
        int print = 0;

        if (argc >= 2) N = std::atoi(argv[1]);
        if (argc >= 3) p = std::atof(argv[2]);
        if (argc >= 4) seed = (unsigned)std::atoi(argv[3]);
        if (argc >= 5) repeats = std::atoi(argv[4]);
        if (argc >= 6) verify = std::atoi(argv[5]);
        if (argc >= 7) print = std::atoi(argv[6]);

        if (N <= 0) { std::fprintf(stderr, "ERROR: N debe ser > 0\n"); return EXIT_FAILURE; }
        if (p < 0.0 || p > 1.0) { std::fprintf(stderr, "ERROR: p debe estar en [0,1]\n"); return EXIT_FAILURE; }
        if (repeats <= 0) repeats = 1;

        const int PRINT_LIMIT = 16;
        if (N <= PRINT_LIMIT) { verify = 1; print = 1; }

        uint8_t* Ain = alloc_matrix(N);
        init_random(Ain, N, p, seed);

        int rc = run_experiment(Ain, N, p, seed, repeats, verify, print);
        std::free(Ain);
        return rc;
    }

    // --- Modo menu ---
    menu_loop();
    return 0;
}
