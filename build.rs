fn main() {
    cc::Build::new()
        .file("./cava/cavacore.c")
        .compile("cavacore");
}
