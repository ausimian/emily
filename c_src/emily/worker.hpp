// WorkerThread: dedicated OS thread owning an MLX stream.
//
// MLX uses thread-local CommandEncoders — a stream's encoder only
// exists on the thread that created it. BEAM processes migrate
// between OS threads, so we pin each MLX stream to a dedicated
// thread and dispatch work to it via run_sync (promise/future) or
// run_async (fire-and-forget, with the task posting its own reply
// via enif_send — see emily/async.hpp).

#pragma once

#include <mlx/mlx.h>

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <type_traits>

namespace emily {

namespace mx = mlx::core;

class WorkerThread {
public:
  WorkerThread() {
    thread_ = std::thread(&WorkerThread::run, this);
    std::unique_lock<std::mutex> lock(mtx_);
    cv_.wait(lock, [this] { return ready_; });
  }

  ~WorkerThread() { stop(); }

  // Enqueue a task without blocking. The caller is responsible for
  // whatever side-effect the task performs (typically enif_send back
  // to a caller PID captured at NIF entry; see emily/async.hpp).
  // Exceptions thrown by the task are swallowed — the task owns
  // error propagation because there is no future to carry an
  // exception through.
  template <typename F>
  void run_async(F &&f) {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      if (stop_)
        throw std::runtime_error("worker thread has been stopped");
      queue_.push([f = std::forward<F>(f), this]() mutable {
        try {
          f(stream_);
        } catch (...) {
          // Swallow — the task owns error propagation.
        }
      });
    }
    cv_.notify_one();
  }

  template <typename F>
  auto run_sync(F &&f) -> decltype(f(std::declval<mx::Stream &>())) {
    using R = decltype(f(std::declval<mx::Stream &>()));
    auto p = std::make_shared<std::promise<R>>();
    auto fut = p->get_future();
    {
      std::lock_guard<std::mutex> lock(mtx_);
      if (stop_)
        throw std::runtime_error("worker thread has been stopped");
      queue_.push([p, f = std::forward<F>(f), this]() mutable {
        try {
          if constexpr (std::is_void_v<R>) {
            f(stream_);
            p->set_value();
          } else {
            p->set_value(f(stream_));
          }
        } catch (...) {
          p->set_exception(std::current_exception());
        }
      });
    }
    cv_.notify_one();
    return fut.get();
  }

  void stop() {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      if (stop_)
        return;
      stop_ = true;
    }
    cv_.notify_one();
    if (thread_.joinable())
      thread_.join();
  }

  mx::Stream stream() const { return stream_; }

private:
  void run() {
    stream_ = mx::new_stream(mx::Device(mx::Device::DeviceType::gpu));
    {
      std::lock_guard<std::mutex> lock(mtx_);
      ready_ = true;
    }
    cv_.notify_one();

    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
        if (queue_.empty())
          break;
        task = std::move(queue_.front());
        queue_.pop();
      }
      task();
    }
  }

  std::thread thread_;
  std::queue<std::function<void()>> queue_;
  std::mutex mtx_;
  std::condition_variable cv_;
  bool stop_ = false;
  bool ready_ = false;
  mx::Stream stream_{0, mx::Device(mx::Device::DeviceType::gpu)};
};

} // namespace emily
