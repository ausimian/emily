defmodule Emily.NifArtifactTest do
  @moduledoc """
  Unit tests for the pure verification helpers behind the precompiled-NIF
  download. The fetch/extract path in `mix.exs` only runs on the
  hex-consumer side (CI builds from source), so these tests are the
  coverage for the checksum-pinning and tar-allowlist logic.
  """

  use ExUnit.Case, async: true

  alias Emily.NifArtifact

  describe "parse_checksums/1" do
    test "parses sha256sum lines, ignoring blanks and comments" do
      contents = """
      # pinned checksums
      ABC123  emily-nif-1.0.0-aot-macos-arm64.tar.gz

      def456  emily-nif-1.0.0-jit-macos-arm64.tar.gz
      """

      assert NifArtifact.parse_checksums(contents) == %{
               "emily-nif-1.0.0-aot-macos-arm64.tar.gz" => "abc123",
               "emily-nif-1.0.0-jit-macos-arm64.tar.gz" => "def456"
             }
    end
  end

  describe "expected/2" do
    test "returns the pinned checksum or :error" do
      m = %{"a.tar.gz" => "deadbeef"}
      assert NifArtifact.expected(m, "a.tar.gz") == {:ok, "deadbeef"}
      assert NifArtifact.expected(m, "missing.tar.gz") == :error
    end
  end

  describe "sha256_hex/1" do
    test "returns lowercase hex (known vector for \"abc\")" do
      assert NifArtifact.sha256_hex("abc") ==
               "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    end
  end

  describe "verify_entries/2" do
    test "accepts exactly the allowlisted regular files" do
      assert NifArtifact.verify_entries([
               {"libemily.so", :regular},
               {"mlx.metallib", :regular}
             ]) == :ok
    end

    test "rejects a non-regular (symlink/hardlink) entry" do
      assert {:error, msg} = NifArtifact.verify_entries([{"libemily.so", :symlink}])
      assert msg =~ "non-regular"
    end

    test "rejects an unexpected entry name" do
      assert {:error, msg} = NifArtifact.verify_entries([{"evil.sh", :regular}])
      assert msg =~ "unexpected"
    end

    test "rejects path traversal and absolute paths (never allowlisted)" do
      assert {:error, _} = NifArtifact.verify_entries([{"../../etc/passwd", :regular}])
      assert {:error, _} = NifArtifact.verify_entries([{"/etc/cron.d/x", :regular}])
      assert {:error, _} = NifArtifact.verify_entries([{"subdir/mlx.metallib", :regular}])
    end
  end
end
