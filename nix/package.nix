{ rustPlatform
, fftw
, llvmPackages
, stdenv
, lib
, rust-bin
, cargo-release
, just
, ...
}:
let
  cargoToml = builtins.fromTOML (builtins.readFile ../Cargo.toml);
  rust-toolchain = rust-bin.fromRustupToolchainFile ../rust-toolchain.toml;
in
rustPlatform.buildRustPackage {
  pname = cargoToml.package.name;
  version = cargoToml.package.version;

  src = builtins.path {
    path = ../.;
  };

  nativeBuildInputs = [
    llvmPackages.clang
    fftw

    # devshell packages
    rust-toolchain
    cargo-release
    just
  ];

  cargoLock.lockFile = ../Cargo.lock;

  LD_LIBRARY_PATH = lib.makeLibraryPath [
    fftw
  ];

  # wtf?
  # https://slightknack.dev/blog/nix-os-bindgen/
  LIBCLANG_PATH = lib.makeLibraryPath [ llvmPackages.libclang.lib ];

  configurePhase = ''
    BINDGEN_CFLAGS="$(< ${stdenv.cc}/nix-support/libc-crt1-cflags) \
      $(< ${stdenv.cc}/nix-support/libc-cflags) \
      $(< ${stdenv.cc}/nix-support/cc-cflags) \
      $(< ${stdenv.cc}/nix-support/libcxx-cxxflags) \
      ${lib.optionalString stdenv.cc.isClang "-idirafter ${stdenv.cc.cc.lib}/lib/clang/${lib.getVersion stdenv.cc.cc}/include"} \
      ${lib.optionalString stdenv.cc.isGNU "-isystem ${lib.getDev stdenv.cc.cc}/include/c++/${lib.getVersion stdenv.cc.cc} -isystem ${stdenv.cc.cc}/include/c++/${lib.getVersion stdenv.cc.cc}/${stdenv.hostPlatform.config}"} \
      $NIX_CFLAGS_COMPILE"
    export OUT=${placeholder "out"}
    echo $OUT
  '';
}
