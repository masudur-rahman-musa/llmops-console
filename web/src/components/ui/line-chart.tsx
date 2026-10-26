const WIDTH = 600;
const HEIGHT = 200;
const PAD = 8;
const LEFT = 48;
const TOP = 20;
const BOTTOM = 40;
const DATE_GAP = 48;

type Point = readonly [number, number];

function scale(value: number, min: number, max: number, from: number, to: number): number {
  return max === min ? (from + to) / 2 : from + ((value - min) / (max - min)) * (to - from);
}

export type Marker = { at: number; label: string };

/** `Sep 15`, in UTC so the server's zone never moves a run to another day. */
function dateLabel(seconds: number): string {
  return new Date(seconds * 1000).toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
}

/** `06:00`, the run's UTC time, printed under its date. */
function timeLabel(seconds: number): string {
  return new Date(seconds * 1000).toISOString().slice(11, 16);
}

/** A marker label at the chart's left or right end is anchored inward so the edge never cuts it. */
function markerAnchor(x: number): "start" | "middle" | "end" {
  if (x <= LEFT) return "start";
  if (x >= WIDTH - PAD) return "end";
  return "middle";
}

/**
 * A server-rendered line chart: no client script, no library.
 * Non-finite points are dropped first; nothing left renders `No data.`. The
 * axis is `<path>` and `<text>` only, so each marker is the only `<line>` it
 * draws; a date closer than 48 px to the last one shown is skipped, and each date
 * shown has its UTC time under it. The svg has no fixed size, so the `chart` class
 * fills the page width. Pinned by tests/chart.test.tsx.
 */
export function LineChart({
  title,
  points,
  markers = [],
  yFormat = String,
}: {
  title: string;
  points: ReadonlyArray<Point>;
  markers?: ReadonlyArray<Marker>;
  yFormat?: (value: number) => string;
}) {
  const kept = points
    .filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y))
    .toSorted((a, b) => a[0] - b[0]);
  if (kept.length === 0) {
    return (
      <figure>
        <figcaption>{title}</figcaption>
        <p className="note">No data.</p>
      </figure>
    );
  }
  const shown = markers.filter((marker) => Number.isFinite(marker.at));
  const xs = [...kept.map(([x]) => x), ...shown.map((marker) => marker.at)];
  const ys = kept.map(([, y]) => y);
  const [minX, maxX] = [Math.min(...xs), Math.max(...xs)];
  const [minY, maxY] = [Math.min(...ys), Math.max(...ys)];
  const at = ([x, y]: Point): [number, number] => [
    scale(x, minX, maxX, LEFT, WIDTH - PAD),
    scale(y, minY, maxY, HEIGHT - BOTTOM, TOP),
  ];
  const coords = kept.map(at);
  const ticks = [...new Set([minY, (minY + maxY) / 2, maxY])];
  let lastDate = Number.NEGATIVE_INFINITY;
  const dates = kept.flatMap(([seconds], index) => {
    const x = coords[index][0];
    if (x - lastDate < DATE_GAP) return [];
    lastDate = x;
    return [{ x, text: dateLabel(seconds), time: timeLabel(seconds) }];
  });
  return (
    <figure>
      <figcaption>{title}</figcaption>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="chart" role="img" aria-label={title}>
        {ticks.map((value) => (
          <text key={`y${value}`} x={LEFT - 6} y={scale(value, minY, maxY, HEIGHT - BOTTOM, TOP) + 4} textAnchor="end" className="chart-text">
            {yFormat(value)}
          </text>
        ))}
        <path d={`M${LEFT} ${TOP} V${HEIGHT - BOTTOM} H${WIDTH - PAD}`} fill="none" className="chart-axis" />
        {dates.map(({ x, text, time }, index) => (
          <g key={`d${index}`}>
            <text x={x} y={HEIGHT - 22} textAnchor="middle" className="chart-text">
              {text}
            </text>
            <text x={x} y={HEIGHT - 8} textAnchor="middle" className="chart-text">
              {time}
            </text>
          </g>
        ))}
        {shown.map((marker, index) => {
          const x = scale(marker.at, minX, maxX, LEFT, WIDTH - PAD);
          return (
            <g key={`m${index}`}>
              <line x1={x} x2={x} y1={TOP} y2={HEIGHT - BOTTOM} className="chart-marker" strokeDasharray="4 4" />
              <text x={x} y={TOP - 6} textAnchor={markerAnchor(x)} className="chart-text">
                {marker.label}
              </text>
            </g>
          );
        })}
        {coords.length > 1 && (
          <polyline points={coords.map(([x, y]) => `${x},${y}`).join(" ")} fill="none" stroke="currentColor" />
        )}
        {coords.map(([x, y], index) => (
          <circle key={`p${index}`} cx={x} cy={y} r={3} fill="currentColor" />
        ))}
      </svg>
    </figure>
  );
}
