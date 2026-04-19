// Async NIF machinery. See docs/planning/async-worker-exploration.md
// for the design rationale.
//
// The pattern: a NIF captures the caller PID via enif_self, mints a
// fresh ref in a process-independent env, enqueues a task onto the
// target WorkerThread, and returns the ref synchronously. The worker
// thread runs the task, builds the reply term in the msg_env, and
// posts {ref, {:ok, payload}} or {ref, {:error, reason}} back to
// the caller via enif_send. The Elixir side awaits the reply with a
// pattern-match receive in `Emily.Native.Async.call/1`.
//
// Three invariants kept by this code:
//
// 1. enif_self must be called on the scheduler thread. The NIF
//    captures the ErlNifPid by value into the lambda; the worker
//    thread (non-scheduler) must never call enif_self itself.
//
// 2. enif_send with a non-NULL msg_env does not transfer env
//    ownership. The worker calls enif_free_env after the send
//    unconditionally — success invalidates the env but the caller
//    still owns the env object.
//
// 3. Resource terms built via enif_make_resource(msg_env, ptr)
//    internally bump the resource's refcount. ResourcePtrs captured
//    by the lambda release their refs on lambda exit, but the term
//    held by msg_env keeps the resource alive until the term is
//    delivered + GC'd on the receiver or msg_env is freed (if the
//    receiver PID is dead).

#pragma once

#include "worker.hpp"

#include <fine.hpp>
#include <mlx/mlx.h>

#include <cstring>
#include <exception>
#include <stdexcept>
#include <utility>

namespace emily {

namespace mx = mlx::core;

namespace __async {

// Build a binary term in msg_env from a null-terminated C string.
inline ERL_NIF_TERM make_binary_from_cstr(ErlNifEnv *msg_env, const char *s) {
  size_t len = std::strlen(s);
  ERL_NIF_TERM term;
  unsigned char *data = enif_make_new_binary(msg_env, len, &term);
  std::memcpy(data, s, len);
  return term;
}

// Build an error reply term in msg_env, classifying the exception
// type. Matches fine::nif_impl's sync catch ladder so the Elixir
// side can raise the same exception classes:
//   std::invalid_argument -> {:argument, message}
//   std::runtime_error    -> {:runtime, message}
//   std::exception        -> {:runtime, message}
//   ... (any)             -> :unknown
inline ERL_NIF_TERM
error_reason_from_current_exception(ErlNifEnv *msg_env) {
  try {
    throw;  // re-raise the current exception to classify it
  } catch (const std::invalid_argument &e) {
    return enif_make_tuple2(msg_env, enif_make_atom(msg_env, "argument"),
                            make_binary_from_cstr(msg_env, e.what()));
  } catch (const std::runtime_error &e) {
    return enif_make_tuple2(msg_env, enif_make_atom(msg_env, "runtime"),
                            make_binary_from_cstr(msg_env, e.what()));
  } catch (const std::exception &e) {
    return enif_make_tuple2(msg_env, enif_make_atom(msg_env, "runtime"),
                            make_binary_from_cstr(msg_env, e.what()));
  } catch (...) {
    return enif_make_atom(msg_env, "unknown");
  }
}

}  // namespace __async

// Run `build_payload` on the worker thread of `w` and post the
// result back to the caller PID as a message. Returns a fresh ref
// synchronously; the caller awaits the reply via
// `Emily.Native.Async.call/1`.
//
// `build_payload` signature:
//     (mx::Stream &stream, ErlNifEnv *msg_env) -> ERL_NIF_TERM
//
// The returned ERL_NIF_TERM is the *payload* term; this helper wraps
// it as `{ref, {:ok, payload}}`. Exceptions thrown by the lambda are
// caught and posted as `{ref, {:error, reason}}` where `reason` is
// `{:argument | :runtime, binary}` or `:unknown`.
template <typename BuildPayload>
fine::Term async_reply(ErlNifEnv *env,
                       fine::ResourcePtr<WorkerThread> w,
                       BuildPayload &&build_payload) {
  ErlNifPid caller;
  enif_self(env, &caller);

  // Mint the ref in a durable (process-independent) env so it
  // survives the NIF return. Copy into the caller's env for the
  // synchronous return value.
  ErlNifEnv *msg_env = enif_alloc_env();
  ERL_NIF_TERM ref_in_msg = enif_make_ref(msg_env);
  ERL_NIF_TERM ref_to_return = enif_make_copy(env, ref_in_msg);

  try {
    w->run_async([msg_env, ref_in_msg, caller,
                  build_payload = std::forward<BuildPayload>(build_payload)]
                 (mx::Stream &s) mutable {
      ERL_NIF_TERM reply;
      try {
        ERL_NIF_TERM payload = build_payload(s, msg_env);
        ERL_NIF_TERM ok_tuple = enif_make_tuple2(
            msg_env, enif_make_atom(msg_env, "ok"), payload);
        reply = enif_make_tuple2(msg_env, ref_in_msg, ok_tuple);
      } catch (...) {
        reply = enif_make_tuple2(
            msg_env, ref_in_msg,
            enif_make_tuple2(
                msg_env, enif_make_atom(msg_env, "error"),
                __async::error_reason_from_current_exception(msg_env)));
      }

      // enif_send invalidates msg_env on success but does not take
      // ownership of the env object itself; free it unconditionally.
      enif_send(nullptr, &caller, msg_env, reply);
      enif_free_env(msg_env);
    });
  } catch (...) {
    // Enqueue failed (worker stopped). Reclaim the env and rethrow
    // so fine::nif_impl surfaces the exception synchronously.
    enif_free_env(msg_env);
    throw;
  }

  return fine::Term(ref_to_return);
}

// Convenience wrapper over async_reply for lambdas that return a
// fine-encodable value (typically a `fine::ResourcePtr<Tensor>` or
// a tuple of them). The helper wraps the computation, encodes the
// result in msg_env, and posts `{ref, {:ok, encoded}}` back.
//
// `build` signature:
//     (mx::Stream &stream) -> T   (T is any fine-encodable type)
template <typename F>
fine::Term async_encoded(ErlNifEnv *env,
                         fine::ResourcePtr<WorkerThread> w,
                         F &&build) {
  return async_reply(
      env, w,
      [build = std::forward<F>(build)](mx::Stream &s, ErlNifEnv *msg_env) mutable {
        return fine::encode(msg_env, build(s));
      });
}

}  // namespace emily
