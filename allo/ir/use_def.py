# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=unused-argument

import ast
import inspect
import textwrap

from .symbol_resolver import ASTResolver
from .utils import get_func_id_from_param_types, resolve_generic_types


class VarNode:
    def __init__(self, path, name, idx=None):
        self.path = path
        self.name = name
        self.users = set()
        # Used for identifying the location of function arguments
        # if self.idx is None, this variable is not a function argument
        self.idx = idx

    def add_user(self, node):
        # node is a VarNode
        self.users.add(node)

    def add_users(self, nodes):
        for node in nodes:
            self.users.add(node)

    def __repr__(self):
        return (
            f"VarNode({self.path}:{self.name})"
            if self.idx is None
            else f"VarNode({self.path}:{self.name}:{self.idx})"
        )


# `class ast.NodeVisitor` is a node visitor base class that walks the abstract 
# syntax tree and calls a visitor function for every node found. This function 
# may return a value which is forwarded by the `visit()` method. This class is 
# meant to be subclassed, with the subclass adding visitor methods such as ex-
# tracting variable names, functions, and calls.. Per default the visitor func-
# tions for the nodes are ``'visit_'`` + class name of the node. 
# So a `TryFinally` node visit function would be `visit_TryFinally`. This beha-
# vior can be changed by overriding the `visit` method. If no visitor function 
# exists for a node (return value `None`) the `generic_visit` visitor is used
# instead. Don't use the `NodeVisitor` if you want to apply changes to nodes 
# during traversing. For this a special visitor exists (`NodeTransformer`) that
# allows modifications.
class UseDefChain(ast.NodeVisitor):
    def __init__(self, global_vars, instantiate):
        self.buffers = {}
        self.path = ""
        self.global_vars = global_vars
        self.inst = instantiate
        # Used for nested functions
        self.arg_nodes = []
        # Used for unique function identification when calling the same function
        # Need to consider different situations:
        # 1) Explicit name is not given in the parameter list
        #    Gernerate a unique name for each function call based on whether the instatiation has been seen before
        # 2) Explicit name is given in the parameter list
        #    Use the explicit name
        # name -> id -> (param_type1, param_type2, ...)
        self.func_name2id = {}
        self.func_id = None
        # Used for metaprogramming
        self.meta_if_stack = []

    def __getitem__(self, key):
        return self.buffers[key]

    def get_name(self, name):
        if self.path == "":
            return name
        return self.path + ":" + name

    def dump_graph(self, top_func_name):
        DEBUG = 0
        if DEBUG:
            print("type of self.buffers: ", type(self.buffers))
            for item,key in self.buffers.items():
                print(f"  {item:>20} = {key}")
        print("digraph G {")
        for var in self.buffers.values():
            var_path = var.path
            if var.path == top_func_name:
                print(f"  {var_path}_{var.name} [style=filled, color=gray];")
            users = ", ".join([f"{user.path}_{user.name}" for user in var.users])
            print(f"  {var_path}_{var.name} -> {{{users}}}")
        print("}")

    def get_equivalent_tensors(self, target_key):
        def recursive_helper(key):
            local_res = []
            path = key.split(":")[0]
            for tensor in self.buffers[key].users:
                if tensor.path != path:
                    local_res.append(tensor)
                    local_res += recursive_helper(tensor.path + ":" + tensor.name)
            return local_res

        results = {key: set() for key in self.buffers}
        for key, buffer in self.buffers.items():
            res = recursive_helper(key)
            results[key].update(set(res))
            for tensor in res:
                results[f"{tensor.path}:{tensor.name}"].add(buffer)
        return results[target_key] if target_key in results else set()

    def visit_Constant(self, node):
        return []

    def visit_Name(self, node):
        if self.get_name(node.id) in self.buffers:
            return set([self.buffers[self.get_name(node.id)]])
        return set()

    def visit_Attribute(self, node):
        return self.visit(node.value)

    def visit_Index(self, node):
        return self.visit(node.value)

    def visit_Tuple(self, node):
        res = []
        for elt in node.elts:
            res += list(self.visit(elt))
        return set(res)

    def visit_BoolOp(self, node):
        return set()

    def visit_List(self, node):
        return set()

    def visit_UnaryOp(self, node):
        return self.visit(node.operand)

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        return set(left).union(set(right))

    def visit_Compare(self, node):
        left = self.visit(node.left)
        right = self.visit(node.comparators[0])
        return set(left).union(set(right))

    def visit_IfExp(self, node):
        cond = self.visit(node.test)
        if_branch = self.visit(node.body)
        else_branch = self.visit(node.orelse)
        return set(cond).union(set(if_branch)).union(set(else_branch))

    def visit_For(self, node):
        # Here, node is a `<ast.For object>`.
        # `<ast.For object>.orelse` means the Python grammer like:
        #     for x in iterable:
        #         # Loop body
        #         ...
        #     else:
        #         # Else block
        #         ...
        # The `for` loop iterates over an iterable (which can be a list, tuple, 
        # dictionary, set, string, etc.). In each iteration, the current element 
        # is assigned to `x`. If the `for` loop completes normally (i.e., it does 
        # not encounter a `break` statement), the code in the `else` block is exe-
        # cuted. If the `for` loop is terminated prematurely using `break`, the 
        # `else` block will not be executed.
        if node.orelse:
            # If the `orelse` attribute exists, a runtime error is thrown, and it 
            # is prompted that the `else` grammer are not supported in Allo.
            raise RuntimeError("'else' clause for 'for' not supported in Allo kernels")
        
        # print(f"isinstance(node.iter, ast.Call): {isinstance(node.iter, ast.Call)}")

        # Check whether the `node.iter` (iterative object) of the `for` loop is a 
        # function call. `ast.Call` means that this is a function call node. The 
        # form of the `for` loop that meets this condition includes: using `range`, 
        # or other functions that generate iterative objects.
        if isinstance(node.iter, ast.Call):
            # `ASTResolver.resolve` resolves a given AST node to a Python object.
            # Here, it transforms a <ast.Name object> to a Python object like None.
            obj = ASTResolver.resolve(node.iter.func, self.global_vars)
            if (
                obj is None
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"
            ) or (obj is not None and obj.__name__ in {"grid", "reduction"}):
                res = []
                for stmt in node.body:
                    res.append(self.visit(stmt))
                return res
        raise RuntimeError("Unsupported for loop")

    def visit_Call(self, node):
        print(f"node: {node}")
        print(f" self.func_id: {self.func_id}")
        print(f" node.func.id: {node.func.id}")
        print(f" node.func: {node.func}")
        # Here, node is a `<ast.Call object>`.
        original_func_id = self.func_id
        if isinstance(node.func, ast.Name):
            # `ASTResolver.resolve` resolves a given AST node to a Python object. 
            # Here, it transforms a <ast.Name object> to a Python <function ...>
            # object.
            obj = ASTResolver.resolve(node.func, self.global_vars)
            # `node.func.id` returns the name of function that is called.
            obj_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            obj = ASTResolver.resolve(node.func, self.global_vars)
            obj_name = node.func.attr
        elif isinstance(node.func, ast.Subscript):
            obj = ASTResolver.resolve(node.func.value, self.global_vars)
            assert obj is not None, "Unsupported function call"
            obj_name = node.func.value.id
            self.inst = ASTResolver.resolve_param_types(
                node.func.slice, self.global_vars
            )
            if self.func_id is None:
                func_id = get_func_id_from_param_types(self.inst)
                if func_id is None:
                    func_dict = self.func_name2id.setdefault(obj_name, {})
                    for key, value in func_dict.items():
                        if value == tuple(self.inst):
                            func_id = key
                            break
                    else:
                        func_id = len(func_dict) if len(func_dict) > 0 else None
                        func_dict[func_id] = tuple(self.inst)
                else:
                    self.inst.remove(func_id)
                    func_dict = self.func_name2id.setdefault(obj_name, {})
                    func_dict[func_id] = tuple(self.inst)
                self.func_id = func_id
        else:
            raise RuntimeError("Unsupported function call")

        if obj is None:
            if isinstance(node.func, ast.Attribute):
                # x.T or x.reverse
                return self.visit(node.func.value)
            if node.func.id in {"float", "int"}:
                # Python-Builtin functions
                return list(self.visit(node.args[0]))
            raise RuntimeError(f"Unsupported function call {node.func.id}")

        if obj.__module__.startswith("allo") and not obj.__module__.startswith(
            "allo.library"
        ):
            arg_nodes = []
            for arg in node.args:
                arg_nodes += list(self.visit(arg))
            return arg_nodes

        # User-defined subfunction
        func = self.global_vars[obj_name]
        arg_nodes = []
        # The arguments have order
        for arg in node.args:
            arg_nodes += list(self.visit(arg))
        if isinstance(func, ast.FunctionDef):
            # Has already been defined in the top-level scope
            tree = func
        else:
            src, _ = inspect.getsourcelines(func)
            src = [textwrap.fill(line, tabsize=4, width=9999) for line in src]
            src = textwrap.dedent("\n".join(src))
            tree = ast.parse(src)
        original_arg_nodes = self.arg_nodes
        self.arg_nodes = arg_nodes
        ret = self.visit(tree)
        if ret is not None:
            arg_nodes += list(ret)
        self.arg_nodes = original_arg_nodes
        self.func_id = original_func_id
        return arg_nodes

    def visit_Assign(self, node):
        # Compute RHS
        if len(node.targets) > 1:
            raise NotImplementedError(
                "Multiple assignment in one statement not supported"
            )
        parents = self.visit(node.value)

        def get_name(subnode):
            if hasattr(subnode, "id"):
                return subnode.id
            return get_name(subnode.value)

        targets = []
        if isinstance(node.targets[0], ast.Tuple):
            targets = node.targets[0].elts
        else:
            targets = [node.targets[0]]
        for i, target in enumerate(targets):
            name = get_name(target)
            var = VarNode(self.path, name)
            if (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "get_pid"
            ):
                self.global_vars[ast.unparse(target)] = self.global_vars[f"df.p{i}"]
            for parent in parents:
                parent.add_user(var)
            if self.get_name(name) not in self.buffers:
                self.buffers[self.get_name(name)] = var

    def visit_AnnAssign(self, node):
        var = VarNode(self.path, node.target.id)
        if node.value is not None:
            parents = self.visit(node.value)
            for parent in parents:
                parent.add_user(var)
        if self.get_name(node.target.id) not in self.buffers:
            self.buffers[self.get_name(node.target.id)] = var

    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Subscript):
            name = node.target.value.id
        elif isinstance(node.target, ast.Name):  # scalar
            name = node.target.id
        else:
            raise NotImplementedError("Unsupported AugAssign")
        var = VarNode(self.path, name)
        parents = self.visit(node.value)
        for parent in parents:
            parent.add_user(var)
        if self.get_name(name) not in self.buffers:
            self.buffers[self.get_name(name)] = var

    def visit_Subscript(self, node):
        res = self.visit(node.value)
        return res

    def visit_FunctionDef(self, node):
        # Here, node is a `<ast.FunctionDef object>`.
        original_path = self.path
        if self.func_id is None:
            # `node.name` is the function name defined by users.
            self.path = node.name
        else:
            self.path = node.name + "_" + str(self.func_id)

        # Generic function
        if (
            len(self.inst) > 0
            and hasattr(node, "type_params")
            and len(node.type_params) > 0
        ):
            assert len(self.inst) == len(
                node.type_params
            ), f"Type parameters mismatch, got {self.inst} and {node.type_params}"
            for type_var, call_val in zip(node.type_params, self.inst):
                name, call_val = resolve_generic_types(
                    self.global_vars, type_var, call_val
                )
                self.global_vars[name] = call_val

        if original_path == "":  # top-level function
            # create initial variables
            for i, arg in enumerate(node.args.args):
                if self.get_name(arg.arg) not in self.buffers:
                    self.buffers[self.get_name(arg.arg)] = VarNode(
                        node.name, arg.arg, i
                    )
        else:
            for i, (inner_arg, outer_arg) in enumerate(
                zip(node.args.args, self.arg_nodes)
            ):
                if self.get_name(inner_arg.arg) not in self.buffers:
                    self.buffers[self.get_name(inner_arg.arg)] = VarNode(
                        self.path, inner_arg.arg, i
                    )
                outer_arg.add_user(self.buffers[self.get_name(inner_arg.arg)])
        res = []
        for stmt in node.body:
            res.append(self.visit(stmt))
        self.path = original_path
        # Add the visited function to global variable for later reference
        self.global_vars[node.name] = node
        return res[-1]

    def visit_Module(self, node):
        res = []
        assert (
            len(node.body) == 1
        ), "Only one function definition in a module is allowed"
        for stmt in node.body:
            res.append(self.visit(stmt))
        return res[0]

    def visit_With(self, node):
        assert len(node.items) == 1, "Only support one context manager"
        assert isinstance(
            node.items[0].context_expr, ast.Call
        ), "Only support `with allo.meta_if/elif/else()`"
        assert isinstance(
            node.items[0].context_expr.func, ast.Attribute
        ), "Only support `with allo.meta_if/elif/else()`"
        assert (
            len(node.items[0].context_expr.args) <= 1
        ), "Only support one argument for `allo.meta_if/elif/else()`"
        # Compile-time comparison
        if node.items[0].context_expr.func.attr in {"meta_if", "meta_elif"}:
            try:
                # pylint: disable=eval-used
                cond = eval(
                    compile(
                        ast.Expression(node.items[0].context_expr.args[0]), "", "eval"
                    ),
                    self.global_vars,
                )
            # pylint: disable=broad-exception-caught
            except Exception:
                return None
            if node.items[0].context_expr.func.attr == "meta_if":
                final_cond = cond
                self.meta_if_stack.append(final_cond)
            else:  # meta_elif
                assert len(self.meta_if_stack) > 0, "Unmatched allo.meta_elif()"
                if self.meta_if_stack[-1]:  # previous `if` has already satisfied
                    self.meta_if_stack.pop()
                    self.meta_if_stack.append(True)
                    final_cond = False
                else:
                    self.meta_if_stack.pop()
                    self.meta_if_stack.append(cond)
                    final_cond = cond
        elif node.items[0].context_expr.func.attr == "meta_else":
            assert len(self.meta_if_stack) > 0, "Unmatched allo.meta_else()"
            final_cond = not self.meta_if_stack[-1]
            self.meta_if_stack.pop()
        else:
            raise RuntimeError("Unsupported meta function")
        if final_cond:
            res = []
            for stmt in node.body:
                res.append(self.visit(stmt))
            return res[-1]
        return None

    def visit_Return(self, node):
        if node.value is None:
            return []
        results = self.visit(node.value)
        # update labels
        for res in results:
            assert isinstance(res, VarNode)
            res.idx = -1
        return results
