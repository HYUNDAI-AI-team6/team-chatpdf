// 배경 장식: 상단 구름 + 하단 언덕 + 언덕 위를 걷는 캐릭터
// app.py가 window.SCENE_IMAGES = [캐릭터1 이미지, 캐릭터2 이미지, ...] 를 먼저 넣어줌
// 움직이는 이미지(webp/gif)면 그대로 재생, 정지 이미지면 통통 튀며 뒤뚱거리는 효과를 줌
(function () {
  const SVG_NS = "http://www.w3.org/2000/svg";
  const SCENE_H = 110;      // 언덕 영역 높이(px)
  const WALKER_H = 96;      // 캐릭터 높이(px)
  const SPEED = 78;         // 걷는 속도(px/초)
  const STEP = 26;          // 한 걸음 길이(px)

  // 언덕 모양: x좌표 → 언덕 영역 위에서부터의 높이
  const frontHill = (x) => 58 + 13 * Math.sin(x / 190) + 6 * Math.sin(x / 71 + 1.3);
  const backHill = (x) => 40 + 18 * Math.sin(x / 260 + 2.1) + 7 * Math.sin(x / 113);

  function hillPath(fn, w) {
    let d = `M 0 ${SCENE_H} L 0 ${fn(0).toFixed(1)}`;
    for (let x = 8; x <= w + 8; x += 8) d += ` L ${x} ${fn(x).toFixed(1)}`;
    return d + ` L ${w + 8} ${SCENE_H} Z`;
  }

  function cloudSVG(scale) {
    return `<svg width="${140 * scale}" height="${70 * scale}" viewBox="0 0 140 70">
      <path d="M30 60 Q8 60 10 44 Q12 30 30 32 Q32 12 54 14 Q66 0 84 10 Q104 6 108 26 Q130 26 130 44 Q130 60 110 60 Z"
        fill="#F2FBF8" stroke="#CDEFE6" stroke-width="2.5" stroke-linejoin="round"/></svg>`;
  }

  function init() {
    if (document.getElementById("scene-hills")) return;
    const images = (window.SCENE_IMAGES || []).filter(Boolean);
    const animated = /^data:image\/(gif|webp)/.test(images[0] || ""); // 이미지 자체가 걷는 애니메이션인지
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    // 구름
    const sky = document.createElement("div");
    sky.id = "scene-sky";
    [
      { top: 14, scale: 1.05, dur: 80, delay: 0 },
      { top: 58, scale: 0.7, dur: 100, delay: -35 },
      { top: 22, scale: 0.85, dur: 88, delay: -60 },
      { top: 70, scale: 1.15, dur: 118, delay: -85 },
    ].forEach((c) => {
      const el = document.createElement("div");
      el.className = "scene-cloud";
      el.style.top = c.top + "px";
      el.style.animationDuration = c.dur + "s";
      el.style.animationDelay = c.delay + "s";
      el.innerHTML = cloudSVG(c.scale);
      sky.appendChild(el);
    });
    document.body.appendChild(sky);

    // 언덕
    const hills = document.createElementNS(SVG_NS, "svg");
    hills.id = "scene-hills";
    const back = document.createElementNS(SVG_NS, "path");
    const front = document.createElementNS(SVG_NS, "path");
    back.setAttribute("fill", "#DDF3EA");
    front.setAttribute("fill", "#BFE9DA");
    front.setAttribute("stroke", "#9EE3D2");
    front.setAttribute("stroke-width", "2");
    hills.append(back, front);
    document.body.appendChild(hills);

    function drawHills() {
      const w = window.innerWidth;
      hills.setAttribute("width", w);
      hills.setAttribute("height", SCENE_H);
      hills.setAttribute("viewBox", `0 0 ${w} ${SCENE_H}`);
      back.setAttribute("d", hillPath(backHill, w));
      front.setAttribute("d", hillPath(frontHill, w));
    }
    drawHills();
    window.addEventListener("resize", drawHills);

    if (!images.length) return;

    // 캐릭터들: 첫 번째가 앞장서고, 나머지는 GAP 간격으로 뒤따라 걸음
    const GAP = 115;
    const walkers = images.map((src, i) => {
      const el = document.createElement("img");
      el.className = "scene-walker";
      el.src = src;
      el.alt = "";
      document.body.appendChild(el);
      return el;
    });

    let x = window.innerWidth + 40; // 앞장선 캐릭터의 위치
    let last = performance.now();

    function place(now) {
      const dt = Math.min((now - last) / 1000, 0.1);
      last = now;
      const tail = GAP * (walkers.length - 1);
      if (!reduce) {
        x -= SPEED * dt;
        if (x < -WALKER_H - 60 - tail) x = window.innerWidth + 40; // 마지막 캐릭터까지 사라지면 오른쪽에서 다시 등장
      } else {
        x = window.innerWidth * 0.7;
      }

      walkers.forEach((el, i) => {
        const ex = x + GAP * i;
        const ww = el.offsetWidth || WALKER_H * 0.8;
        const footX = ex + ww / 2;
        const groundY = window.innerHeight - SCENE_H + frontHill(footX);
        const slope = Math.atan2(frontHill(footX + 4) - frontHill(footX - 4), 8) * (180 / Math.PI);

        const phase = (window.innerWidth + 40 - ex) / STEP + i * 0.5;       // 캐릭터마다 걸음 박자를 살짝 다르게
        const still = reduce || animated;                                    // 움직이는 이미지는 자체 동작만 사용
        const bob = still ? 0 : -Math.abs(Math.sin(phase * Math.PI)) * 5;   // 걸을 때마다 살짝 뜀
        const waddle = still ? 0 : Math.sin(phase * Math.PI) * 6;          // 뒤뚱뒤뚱
        el.style.left = ex + "px";
        el.style.top = groundY - WALKER_H + 4 + bob + "px";
        el.style.transform = `rotate(${slope * 0.6 + waddle}deg)`;        // 언덕 경사에 맞춰 살짝 기울임
      });
      requestAnimationFrame(place);
    }
    requestAnimationFrame(place);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
