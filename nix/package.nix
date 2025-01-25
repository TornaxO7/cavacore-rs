{ rustPlatform
, glibc
, fftw
, libclang
, llvmPackages
, stdenv
, lib
, ...
}:
let
  cargoToml = builtins.fromTOML (builtins.readFile ../Cargo.toml);
in
rustPlatform.buildRustPackage {
  pname = cargoToml.package.name;
  version = cargoToml.package.version;

  src = builtins.path {
    path = ../.;
  };

  buildInputs = [
    fftw
    libclang
    glibc
  ];

  nativeBuildInputs = [
    llvmPackages.clang
  ];

  cargoLock.lockFile = ../Cargo.lock;

  # wtf?
  # https://slightknack.dev/blog/nix-os-bindgen/

  LIBCLANG_PATH = "${llvmPackages.libclang.lib}/lib";

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
