# Dialyzer warning suppressions. Every entry is a false positive driven
# by a dialyzer limitation documented below, not a correctness issue.
# Use `{file, warning_type[, line]}` tuples; add a comment per cluster.

[
  # ---------------------------------------------------------------
  # Emily.Backend — Nx.Backend callback type mismatches
  # ---------------------------------------------------------------
  #
  # `wrap/3` (lib/emily/backend.ex:71) returns `%T{out | data: %B{...}}`,
  # which dialyzer's map-update analysis infers as a partial map
  # (`%Nx.Tensor{data, type, _ => _}`) — the unchanged `:shape`,
  # `:names`, `:vectorized_axes` keys drop out of the inferred type
  # because dialyzer doesn't propagate struct field types across map
  # updates. Nx.Backend's callback specs require the full
  # `Nx.Tensor.t()` record, so dialyzer reports a "nothing in common"
  # mismatch for every `@impl`'d callback (≈115 warnings).
  #
  # The code IS correct — we genuinely return complete `%Nx.Tensor{}`
  # structs — so suppressing the category wholesale is the right call.
  # Attempts to coax dialyzer (guards, explicit struct reconstruction,
  # narrower `@type tensor`) fix the shape/names tracking but introduce
  # their own quirks. Revisit if dialyzer's map-update analysis improves
  # (pre-Erlang 28 this was worse).
  {"lib/emily/backend.ex", :callback_type_mismatch},
  {"lib/emily/backend.ex", :callback_arg_type_mismatch},
  # Same root cause — `wrap/3`'s success typing ends up wider than the
  # declared `@spec wrap(ref(), tensor(), reference()) :: tensor()`.
  # Line omitted because dialyzer's reported position varies with the
  # inferred typing that flows in from Emily.Native (which became
  # wider when the op NIFs were converted to async).
  {"lib/emily/backend.ex", :invalid_contract},

  # ---------------------------------------------------------------
  # Emily.Fast — `Nx.Defn.Expr.optional/3` untyped return
  # ---------------------------------------------------------------
  #
  # `Nx.Defn.Expr.optional/3` has no `@spec` (see
  # `deps/nx/lib/nx/defn/expr.ex`). Dialyzer infers its return as
  # `tuple() | %{data: %Nx.Defn.Expr{…}, _ => _}` — the tuple branch
  # only fires when the fallback fun itself returns a tuple, which ours
  # don't. At runtime every one of these returns a proper
  # `Nx.Tensor.t()`, matching the declared `@spec`. Suppress the contract
  # check until Nx specs `optional/3`.
  {"lib/emily/fast.ex", :invalid_contract},

  # ---------------------------------------------------------------
  # Emily.Quantization / Emily.QuantizedWeight — tensor construction
  # ---------------------------------------------------------------
  #
  # `quantized_matmul/2`, `from_dense/2`, and `to_dense/1` build a
  # fresh `%T{data: %B{ref: …}, shape: …, type: …, names: …}` from
  # NIF-returned refs. Dialyzer infers the resulting tensor's `:type`
  # as `{atom(), non_neg_integer()}` — `Native.dtype/1`'s declared
  # return — which doesn't overlap with `Nx.Tensor.t()`'s
  # `Nx.Type.t()` specific union. The NIF does return a valid
  # `Nx.Type.t()` tuple; the width is a declaration limitation in
  # the stub module.
  # Line numbers omitted: positions shift with inferred typing from
  # Emily.Native, which widened when op NIFs moved to the async model.
  {"lib/emily/quantization.ex", :invalid_contract},
  {"lib/emily/quantized_weight.ex", :invalid_contract}
]
