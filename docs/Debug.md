# Debugging
This page describes how to debug X-BACH, as well as HOTPANTS, using Visual Studio Code.

## Debugging X-BACH
Build X-BACH by following the normal build instructions. Make sure to build as Debug, in other words with `--config Debug` flag.

Three JSON files are needed in `.vscode` folder. The following files are illustrations of its content.

c_cpp_properties.json:

``` json
{
    "configurations": [
        {
            "name": "Win64",
            "includePath": [
                "${workspaceFolder}",
                "<path to opencl include>",
                "<path to cfitsio include>",
                "<path to ccfits include>"
            ],
            "intelliSenseMode": "windows-msvc-x64",
            "compilerPath": "<path to compiler>",
            "cStandard": "c17",
            "cppStandard": "c++20"
        }
    ],
    "version": 4
}
```

launch.json:

``` json
{
    "configurations": [
        {
            "name": "(Windows) Launch",
            "type": "cppvsdbg",
            "request": "launch",
            "program": "${workspaceFolder}/build/Debug/BACH.exe",
            "args": [],
            "stopAtEntry": false,
            "cwd": "${fileDirname}",
            "environment": [],
            "console": "internalConsole"
        }
    ]
}
```

## Debugging HOTPANTS
Build HOTPANTS as debug, by passing the flags `-O0 -ggdb`. The following two files are needed in the `.vscode`.

c_cpp_properties.json:

``` json
{
    "configurations": [
        {
            "name": "Win64",
            "includePath": [
                "${workspaceFolder}",
                "<path to opencl include>",
                "<path to cfitsio include>"
            ],
            "intelliSenseMode": "windows-msvc-x64",
            "compilerPath": "<path to compiler>",
            "cStandard": "c17",
            "cppStandard": "c++20"
        }
    ],
    "version": 4
}
```

launch.json:

``` json
{
    "configurations": [
        {
            "name": "(gdb) Launch",
            "type": "cppdbg",
            "request": "launch",
            "program": "${workspaceFolder}/HOTPANTS.exe",
            "args": [],
            "stopAtEntry": false,
            "cwd": "${fileDirname}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "miDebuggerPath": "<path to gdb>",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for gdb",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                },
                {
                    "description": "Set Disassembly Flavor to Intel",
                    "text": "-gdb-set disassembly-flavor intel",
                    "ignoreFailures": true
                }
            ]
        }
    ]
}
```
