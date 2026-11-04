# Vector alignment in matrix Lie groups: interactive page

A static site for GitHub Pages. All of the numerical work runs in the browser.

- `src/lialign.c` is compiled to `lialign.wasm`. It contains the Lie algebra method, Gauss-Newton refinement, data generators and diagnostics, and it does not use the C library.
- `src/lialign.wgsl` is a WebGPU compute shader for the Monte Carlo study, with one alignment problem per GPU thread.
- Examples: Lorentz transformations from particle four-momenta (SO(3,1)), superposition of molecules (SE(3)), storage ring one-turn maps (Sp(4)), two-qubit gate identification (U(4)), and a GPU Monte Carlo comparing the Lie algebra estimate with Gauss-Newton in SO(3), SL(3), SO(3,1), Sp(4) and SE(3).
- An in-page test suite of 18 checks tied to results in the paper. Open `index.html?autotest` to run it on load; the page sets `data-tests-done` on `<body>` when it has finished.
- `app.js` holds the interface and plots, and `index.html` and `style.css` hold the page.

## Deploy
The documentation build (`docs/make.jl`) copies this folder into the site as `demo/`, so the page is published with the package documentation on GitHub Pages at `.../FastLieAlign.jl/dev/demo/`. To serve it on its own instead, set **Settings → Pages → Source** to this folder.

## Run locally
The page loads files with `fetch`, so open it over HTTP, not `file://`:

    python3 -m http.server -d web 8000

## Rebuild the WebAssembly
    web/src/build.sh

This needs `clang` with the `wasm32` target and `wasm-ld` (`brew install lld` on macOS).

MIT License.
