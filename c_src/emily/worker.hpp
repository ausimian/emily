// WorkerThread: a dedicated OS thread owning an MLX stream, plus the
// Reaper that joins finished worker threads off the BEAM schedulers.
//
// MLX uses thread-local CommandEncoders — a stream's encoder only exists
// on the thread that created it. BEAM processes migrate between OS
// threads, so we pin each MLX stream to a dedicated thread and dispatch
// work to it via run_async (fire-and-forget, with the task posting its
// own reply via enif_send — see emily/async.hpp).
//
// Lifetime / teardown:
//   - Per-worker mutable state lives in a heap `State` (shared_ptr), so a
//     worker thread safely outlives the BEAM resource object.
//   - On stop the worker *cancels* its queued tasks (each posts an
//     {:error, :stopped} reply) instead of running them, then exits — so
//     a join waits for at most the one in-flight op, never the backlog.
//   - A resource destructor runs during BEAM GC and must never join() on
//     a scheduler. Instead it signals stop and hands the thread to the
//     Reaper: one long-lived thread that joins finished workers
//     off-scheduler and is itself joined in the NIF unload callback
//     (`Reaper::shutdown`, wired in worker_nif.cpp). The Reaper also
//     tracks live workers so unload can stop+join stragglers.

#pragma once

#include <mlx/mlx.h>

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace emily {

namespace mx = mlx::core;

// A queued unit of work. `cancelled == true` means the worker is
// stopping: the task must post its {:error, :stopped} reply (and free its
// env) instead of running the MLX op. Invoked exactly once per task.
using Task = std::function<void(mx::Stream &, bool /*cancelled*/)>;

// Per-worker mutable state. Heap-allocated and shared between the BEAM
// resource and the worker thread so it outlives whichever drops last.
struct State {
  explicit State(std::size_t queue_limit) : cap(queue_limit) {}

  std::mutex mtx;
  std::condition_variable cv;
  std::queue<Task> queue;
  bool stop = false;
  bool ready = false;
  std::size_t cap;
  mx::Stream stream{0, mx::Device(mx::Device::DeviceType::gpu)};
};

inline void signal_stop(State &st) {
  {
    std::lock_guard<std::mutex> lock(st.mtx);
    st.stop = true;
  }
  st.cv.notify_all();
}

// Best-effort: enqueue `fn` to run (and then be destroyed) on the worker
// thread. Used to drop thread-affine resources — e.g. an `mx::compile`
// cache entry, whose erase must happen on the thread whose thread-local
// compiler cache holds it — on the thread that created them. Bypasses the
// queue cap (teardown is tiny and must not be back-pressured). If the
// worker is already stopping, does nothing: the thread's exit destroys its
// thread-local state anyway, so the resource is reclaimed regardless.
inline void post_to_worker(State &st, std::function<void()> fn) {
  {
    std::lock_guard<std::mutex> lock(st.mtx);
    if (st.stop) {
      return;
    }
    st.queue.push([fn = std::move(fn)](mx::Stream &, bool) mutable { fn(); });
  }
  st.cv.notify_one();
}

// Joins worker threads off the BEAM schedulers. Singleton; its thread is
// created lazily on first use and joined in the NIF unload callback.
class Reaper {
public:
  static Reaper &instance() {
    static Reaper reaper;
    return reaper;
  }

  // Track a newly-spawned worker as live. `key` identifies the worker
  // (its State pointer); `st` is a non-owning handle used to signal stop
  // at unload.
  void track(State *key, std::thread thread, std::weak_ptr<State> st) {
    std::lock_guard<std::mutex> lock(mtx_);
    live_.emplace(key, Live{std::move(thread), std::move(st)});
  }

  // The worker is going away (resource GC or explicit close): stop it and
  // hand its thread off to be joined. Non-blocking — no join on the
  // calling (scheduler) thread.
  void retire(State *key) {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      auto it = live_.find(key);
      if (it == live_.end()) {
        return;  // already retired (e.g. double close)
      }
      if (auto st = it->second.st.lock()) {
        signal_stop(*st);
      }
      dying_.push_back(std::move(it->second.thread));
      live_.erase(it);
    }
    cv_.notify_all();
  }

  // NIF unload: stop and join every worker (live + dying), then stop the
  // reaper thread. Blocking is fine here — the VM is tearing down.
  void shutdown() {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      for (auto &entry : live_) {
        if (auto st = entry.second.st.lock()) {
          signal_stop(*st);
        }
        dying_.push_back(std::move(entry.second.thread));
      }
      live_.clear();
      shutdown_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  ~Reaper() { shutdown(); }

private:
  struct Live {
    std::thread thread;
    std::weak_ptr<State> st;
  };

  Reaper() : thread_([this] { run(); }) {}

  void run() {
    while (true) {
      std::vector<std::thread> batch;
      {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return shutdown_ || !dying_.empty(); });
        batch.swap(dying_);
        if (batch.empty() && shutdown_) {
          break;
        }
      }
      for (auto &t : batch) {
        if (t.joinable()) {
          t.join();
        }
      }
    }
  }

  std::mutex mtx_;
  std::condition_variable cv_;
  std::map<State *, Live> live_;
  std::vector<std::thread> dying_;
  bool shutdown_ = false;
  std::thread thread_;
};

class WorkerThread {
public:
  explicit WorkerThread(std::size_t queue_limit)
      : state_(std::make_shared<State>(queue_limit)) {
    std::thread thread([st = state_] { run(st); });
    Reaper::instance().track(state_.get(), std::move(thread), state_);

    std::unique_lock<std::mutex> lock(state_->mtx);
    state_->cv.wait(lock, [this] { return state_->ready; });
  }

  // Non-blocking: signal stop and hand the thread to the Reaper to join
  // off-scheduler. Pending tasks are cancelled with {:error, :stopped}.
  ~WorkerThread() { Reaper::instance().retire(state_.get()); }

  // Enqueue a task. Throws if the worker has been stopped or the queue is
  // at capacity (back-pressure) — both surface synchronously as an
  // exception from the calling NIF.
  void run_async(Task task) {
    {
      std::lock_guard<std::mutex> lock(state_->mtx);
      if (state_->stop) {
        throw std::runtime_error("worker thread has been stopped");
      }
      if (state_->queue.size() >= state_->cap) {
        throw std::runtime_error(
            "worker queue is full (limit " + std::to_string(state_->cap) +
            "); too many operations are queued on this stream");
      }
      state_->queue.push(std::move(task));
    }
    state_->cv.notify_one();
  }

  // Signal stop without blocking. The Reaper joins the thread when the
  // resource is collected; queued tasks are cancelled with :stopped.
  void stop() { signal_stop(*state_); }

  std::size_t queue_depth() {
    std::lock_guard<std::mutex> lock(state_->mtx);
    return state_->queue.size();
  }

  // A non-owning handle to this worker's state, for posting thread-affine
  // teardown (see `post_to_worker`) from an object that outlives — or is
  // collected independently of — the worker resource. Weak so the worker
  // can be reclaimed while the holder lives; `lock()` fails once it is.
  std::weak_ptr<State> weak_state() const { return state_; }

private:
  static void run(std::shared_ptr<State> st) {
    st->stream = mx::new_stream(mx::Device(mx::Device::DeviceType::gpu));
    {
      std::lock_guard<std::mutex> lock(st->mtx);
      st->ready = true;
    }
    st->cv.notify_all();

    while (true) {
      Task task;
      bool cancelled = false;
      {
        std::unique_lock<std::mutex> lock(st->mtx);
        st->cv.wait(lock, [&] { return st->stop || !st->queue.empty(); });
        if (st->queue.empty()) {
          break;  // stop_ set and the queue is drained
        }
        task = std::move(st->queue.front());
        st->queue.pop();
        cancelled = st->stop;
      }
      // Exceptions are owned by the task (it posts its own error reply).
      task(st->stream, cancelled);
    }
  }

  std::shared_ptr<State> state_;
};

} // namespace emily
