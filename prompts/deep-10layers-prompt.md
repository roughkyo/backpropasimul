# Master Prompt — 10-Hidden-Layer Interactive Backpropagation Simulator (KR/EN Hybrid)

> **Purpose**: 3층(입력–은닉 1–출력) 역전파 시뮬레이션을 **은닉층 10개(총 12층: Input → H1..H10 → Output)** 규모로 확장하되, 동일한 글래스모피즘/뉴모피즘 미학·화이트/다크 테마·확대/축소(Zoom & Pan)·광학 돋보기(Loupe)·요소 클릭 해부 원리를 100% 유지하여 구현하기 위한 재사용 가능 프롬프트입니다.

---

```markdown
# Role & Objective
You are a Lead EdTech Engineer & Interactive Data Visualization Specialist building a **zero-build, static HTML/CSS/ES-Module web simulator** for high-school AI/CS learners (고1~2 수준, 수식 최소화, 직관적 시각화 중심).

Extend our existing 3-layer backpropagation simulator (`index.html`, M5) into a standalone **10-Hidden-Layer Deep Neural Network Interactive Simulator (`deep-10layers.html`)** linked from the bottom of `index.html` (`target="_blank" rel="noopener noreferrer"`).

---

# 1. Mathematical Architecture (`js/math/deep-backprop.js`)
Implement a pure, UI-independent calculation engine:
- **Network Topology**:
  - Input layer `L0`: 2 nodes (`i1 = 0.05`, `i2 = 0.10`)
  - Hidden layers `H1` ~ `H10` (default `numHiddenLayers = 10`, dynamically adjustable `1..10` for direct shallow-vs-deep comparison): 2 nodes per layer (`h_{k,1}, h_{k,2}`)
  - Output layer `Out`: 2 nodes (`o1`, `o2`), targets `(0.01, 0.99)`
  - Total trainable weights at 10 hidden layers: **11 weight matrices × 4 = 44 weights** (bias omitted for pedagogical continuity with M5).
- **Activation Toggle (Key Pedagogical Contrast)**:
  - `sigmoid` (default): $\sigma(x) = \frac{1}{1+e^{-x}}$, $\sigma'(x) = \sigma(x)(1-\sigma(x)) \le 0.25$ → clearly demonstrates **기울기 소실 (Vanishing Gradient)** across 10 hidden layers.
  - `relu` (Leaky/Standard ReLU): $\max(0, x)$, derivative $1$ for $x>0$ (small leak `0.05` optional or exact ReLU) → clearly demonstrates how gradient survives back to `H1`!
- **Deterministic Initialization**:
  - Use a seeded PRNG (`mulberry32`) for reproducible initial weights; keep the final output matrix `W_out` aligned with M5 (`[[0.35, 0.40], [0.45, 0.50]]`).
- **Exact Backprop & Layer Diagnostics**:
  - Compute per-layer `net`, `activation`, `delta` ($\delta_l$), per-weight `gradient` ($\partial E/\partial W_l$), `deltaW` ($-\alpha \cdot \text{grad}$), and per-layer aggregate metrics (`meanAbsGrad`, `maxAbsGrad`, `relativeToOutputPct`) so the UI can plot real-time **Layer-wise Gradient Attenuation (층별 기울기 감쇠 차트)**.
- **Mandatory Automated Verification (`scripts/verify_deep_math.mjs`)**:
  - Verify all 44 analytic gradients against **central finite differences** $\frac{E(w+\epsilon)-E(w-\epsilon)}{2\epsilon}$ ($\epsilon = 10^{-6}$) for BOTH `sigmoid` and `relu` with tolerance `< 1e-7`.

---

# 2. Interactive Zoom / Pan / Loupe Canvas Engine (`js/visuals/deep-neural-canvas.js`)
Render the 12-layer network on a **60fps High-DPI HTML5 Canvas 2D** using a virtual world coordinate system (`WORLD_W = 3400, WORLD_H = 620`):
- **Zoom & Pan Navigation (Figma / Maps-grade ergonomics)**:
  1. **Default Fit View**: Auto-fit all 12 layers horizontally so the learner sees the full 10-layer gradient fade at a glance.
  2. **Cursor-Centered Mouse Wheel Zoom (`Fit ~ 300%`)**: Zoom smoothly toward the pointer location (`worldPos` invariant under zoom).
  3. **Click-and-Drag Pan**: Smooth panning when zoomed in, with boundary clamping.
  4. **One-Click Camera Focus Presets**: Smoothly animated camera lerp (`lerp(zoom, panX, panY)`) for `[전체 12층 보기]`, `[앞쪽 층 H1~H3]`, `[중간 층 H4~H7]`, `[뒤쪽 층 H8~Out]`, plus individual layer chips (`H1`~`H10`).
  5. **Interactive Minimap Overlay**: Compact live minimap in the bottom-right corner showing all 12 layers and the current viewport frame; clicking/dragging the minimap moves the camera instantly.
  6. **2.2× X-Ray Optical Magnifier Loupe (`Z` key or toggle button)**: Circular frosted-rim glass lens (`r = 128px`) magnifying 2.2× under the cursor and revealing hidden X-Ray values (`net`, `σ'`, `δ`, `∂E/∂w`).
- **Dynamic Signal & Gradient Animation**:
  - **Forward (Blue `#2563eb` / `#3b82f6`, →)**: Crisp glowing particle streams flowing left-to-right across all 12 layers.
  - **Backward (Orange `#ea580c` / `#f97316`, ←)**: Reverse energy waves flowing right-to-left whose **particle brightness, speed, and edge glow scale with each layer's actual gradient magnitude** (dramatically fading near `H1` under Sigmoid, staying bright all the way to `H1` under ReLU!).
  - **Action Effects**: Firing an update triggers a **10-layer sequential backprop shockwave** sweeping from `Out` back to `H1`, node ripples, and floating delta popups (`-0.0036 ↓`, `오차 감소!`).
- **Full Element Hover & Click-to-Inspect (All 71 Elements)**:
  - Hit-test all 24 nodes, 44 weight edges, and 3 error boxes in world coordinates.
  - Hovering highlights connected upstream/downstream edges and displays a floating glass tooltip.
  - Clicking any element opens a slide-in **Glassmorphic Deep Inspector Drawer** showing that exact layer's live numbers, step-by-step chain calculation, attenuation ratio vs. output layer, and a mini activation curve.

---

# 3. Design System & Keyword-Centric UX (`deep-10layers.html`, `css/style.css`)
- **Theme**: Default **White Mode (`data-theme="light"`)** with instant **Dark Mode (`data-theme="dark"`)** switch; universal **Blue (Forward) / Orange (Backward) / Emerald (Loss Reduction)** palette across both DOM and Canvas.
- **Visual Style**: 96vw ultra-wide layout (`max-width: 1760px`), frosted **Glassmorphism** (`backdrop-filter: blur(26px) saturate(180%)`), tactile **Neumorphism** controls, ambient aurora background orbs.
- **Korean Copywriting (한글 윤문)**: Natural, concise classroom Korean for high-school 1st–2nd graders — zero AI-slop phrasing, formulas tucked inside optional `<details>` blocks.
- **Interactive Dashboard Below Canvas**:
  1. **Real-Time 10-Layer Gradient Attenuation Bar Chart** (11 bars `W1..W11` on log/relative scale showing exactly how much each layer learns under Sigmoid vs. ReLU).
  2. **Keyword Micro-Learning Cards & Controls**: `#깊은신경망(10은닉층)`, `#기울기소실`, `#ReLU해결`, `#학습률·연속학습` + sliders (`은닉층 수 1~10`, `학습률 lr`, `입력 i1/i2`) + action console (`⚡ 1회 역전파 발사`, `🚀 왕복 연속 학습`, `Sigmoid ↔ ReLU 즉시 비교`, `초기화`).

---

# 4. Integration & Verification Checklist
- [ ] Add a clean, high-contrast bottom banner in `index.html` linking to `./deep-10layers.html` with `target="_blank" rel="noopener noreferrer"`.
- [ ] Zero external JS dependencies (pure static ES modules + Canvas 2D + SVG, GitHub Pages ready).
- [ ] Run `node scripts/verify_deep_math.mjs` — all 44 weights match central finite difference (`< 1e-7`) for both `sigmoid` and `relu`.
- [ ] Run `node --check` on all `.js` files — 0 syntax errors.
```
