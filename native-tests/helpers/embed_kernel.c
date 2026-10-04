// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Embedded CPython native kernel loader

/* Embed CPython and register the genuine PyO3 initializer as a built-in. */
#include <Python.h>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

/** Load the real PyO3 initializer, run a CPython probe and finalize it. */
int main(int argc, char **argv) {
    if (argc != 4) {
        fputs("usage: embedded-native LIBRARY PYTHON PROBE\n", stderr);
        return 2;
    }
    void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (library == NULL) {
        fprintf(stderr, "dlopen: %s\n", dlerror());
        return 2;
    }
    void *symbol = dlsym(library, "PyInit_spo_kernel");
    if (symbol == NULL) {
        fprintf(stderr, "dlsym: %s\n", dlerror());
        return 2;
    }
    PyObject *(*initialise_kernel)(void);
    _Static_assert(sizeof(initialise_kernel) == sizeof(symbol), "POSIX function pointer size");
    memcpy(&initialise_kernel, &symbol, sizeof(symbol));
    if (PyImport_AppendInittab("spo_kernel.spo_kernel", initialise_kernel) != 0) {
        fputs("cannot register the native initializer\n", stderr);
        return 2;
    }
    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    PyStatus status = PyConfig_SetBytesString(&config, &config.program_name, argv[2]);
    if (!PyStatus_Exception(status)) {
        status = Py_InitializeFromConfig(&config);
    }
    PyConfig_Clear(&config);
    if (PyStatus_Exception(status)) {
        Py_ExitStatusException(status);
    }
    FILE *probe = fopen(argv[3], "r");
    if (probe == NULL) {
        perror("probe");
        Py_FinalizeEx();
        return 2;
    }
    int result = PyRun_SimpleFileEx(probe, argv[3], 1);
    int finalised = Py_FinalizeEx();
    return result == 0 && finalised == 0 ? 0 : 1;
}
