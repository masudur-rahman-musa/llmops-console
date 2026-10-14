const WIDTH = 600;
const HEIGHT = 160;
const PAD = 8;

type Point = readonly [number, number];

function scale(value: number, min: number, max: number, from: number, to: number): number {
  return max === min ? (from + to) / 2 : from + ((value - min) / (max - min)) * (to - from);
}

/**
 * A server-rendered line chart: no client script, no library (decision B).
 * Non-finite points are dropped first; nothing left renders `No data.`. Each
 * marker is the only `<line>` it draws. Pinned by tests/chart.test.tsx.
 */
export function LineChart({
  title,
  points,
  markers = [],
}: {
  title: string;
  points: ReadonlyArray<Point>;
  markers?: ReadonlyArray<number>;
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
  const shown = markers.filter(Number.isFinite);
  const xs = [...kept.map(([x]) => x), ...shown];
  const ys = kept.map(([, y]) => y);
  const [minX, maxX] = [Math.min(...xs), Math.max(...xs)];
  const [minY, maxY] = [Math.min(...ys), Math.max(...ys)];
  const at = ([x, y]: Point): [number, number] => [
    scale(x, minX, maxX, PAD, WIDTH - PAD),
    scale(y, minY, maxY, HEIGHT - PAD, PAD),
  ];
  const coords = kept.map(at);
  return (
    <figure>
      <figcaption>{title}</figcaption>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} width={WIDTH} height={HEIGHT} className="chart" role="img" aria-label={title}>
        {shown.map((marker, index) => {
          const x = scale(marker, minX, maxX, PAD, WIDTH - PAD);
          return <line key={`m${index}`} x1={x} x2={x} y1={0} y2={HEIGHT} className="chart-marker" strokeDasharray="4 4" />;
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
