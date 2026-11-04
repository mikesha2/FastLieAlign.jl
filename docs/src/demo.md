# Browser demo

An interactive page runs the method in the browser, with the numerical work compiled to
WebAssembly and a WebGPU Monte Carlo study.

[Open the demo](demo/index.html)

It covers Lorentz transformations from particle four-momenta in ``SO(3,1)``, superposition of
molecules in ``SE(3)``, one-turn maps in ``Sp(4)``, two-qubit gate identification in ``U(4)``,
and a GPU comparison of the Lie algebra estimate with its Gauss–Newton refinement in
``SO(3)``, ``SL(3)``, ``SO(3,1)``, ``Sp(4)`` and ``SE(3)``. The WebGPU part needs a browser
with WebGPU enabled.

The source is in the `web/` folder of the repository: `src/lialign.c` compiled to
`lialign.wasm`, the shader `src/lialign.wgsl`, and `app.js` for the interface. See
`web/README.md` for rebuilding the WebAssembly.
