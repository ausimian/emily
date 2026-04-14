# emily

Elixir bindings and Nx backend for Apple's [MLX](https://github.com/ml-explore/mlx).

**Status: M0 — scaffold.** See [`PLAN.md`](PLAN.md) for the full roadmap.

## Why

To run Bumblebee models (notably Qwen3) on Apple Silicon with Metal
acceleration, via a layered architecture that keeps each layer
independently testable and avoids the [nif_call-deadlock
class](https://github.com/elixir-nx/emlx/issues/88) that grounded EMLX.

## Requirements

- macOS (Apple Silicon recommended; x86_64 supported)
- Elixir 1.18+ / OTP 27+ (development pinned to 1.19.5 / OTP 28 via `.tool-versions`)

## Usage (M0)

Only a tiny tensor round-trip is wired up today:

```elixir
bin = <<1.0::float-32-native, 2.0::float-32-native, 3.0::float-32-native>>
t = Emily.from_binary(bin, [3], {:f, 32})
Emily.to_binary(t) == bin
Emily.shape(t) == [3]
Emily.dtype(t) == {:f, 32}
```

Full `Nx.Backend` support lands in M2; see the plan.

## License

MIT
