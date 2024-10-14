# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# A set of CMake commands used to automate the build process for an out-of-tree
# MLIR dialect called "Allo". It leverages MLIR's code generation utilities to 
# create C++ header and source files based on TableGen definitions.

# Specifies the TableGen definition file `AlloOps.td`. This file contains the 
# definitions of operations specific to the Allo dialect.
set(LLVM_TARGET_DEFINITIONS AlloOps.td)
# `mlir_tablegen` generates operation declarations from `AlloOps.td` into the 
# header file `AlloOps.h.inc`.
mlir_tablegen(AlloOps.h.inc -gen-op-decls)
# `mlir_tablegen` produces operation definitions from `AlloOps.td` into the 
# source file `AlloOps.cpp.inc`.
mlir_tablegen(AlloOps.cpp.inc -gen-op-defs)
# `mlir_tablegen` generates declarations for the Allo dialect itself into 
# `AlloDialect.h.inc`, specifying the dialect's name.
mlir_tablegen(AlloDialect.h.inc -gen-dialect-decls -dialect=allo)
# `mlir_tablegen` generates the corresponding definitions for the dialect into 
# `AlloDialect.cpp.inc`.
mlir_tablegen(AlloDialect.cpp.inc -gen-dialect-defs -dialect=allo)
# Defines a CMake target (`MLIRAlloOpsIncGen`) responsible for ensuring that 
# the operation-related TableGen files are generated when needed. This helps 
# manage dependencies in the build system.
add_public_tablegen_target(MLIRAlloOpsIncGen)
# Ensures that the `MLIRAlloOpsIncGen` target (operations) is built before the 
# `mlir-headers` target, maintaining proper build order.
add_dependencies(mlir-headers MLIRAlloOpsIncGen)

set(LLVM_TARGET_DEFINITIONS AlloTypes.td)
mlir_tablegen(AlloTypes.h.inc -gen-typedef-decls)
mlir_tablegen(AlloTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRAlloTypesIncGen)

set(LLVM_TARGET_DEFINITIONS AlloAttrs.td)
mlir_tablegen(AlloAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(AlloAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRAlloAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS AlloAttrs.td)
mlir_tablegen(AlloEnums.h.inc -gen-enum-decls)
mlir_tablegen(AlloEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRAlloEnumsIncGen)

add_mlir_doc(AlloDialect AlloDialect Allo/ -gen-dialect-doc)
add_mlir_doc(AlloOps AlloOps Allo/ -gen-op-doc)
